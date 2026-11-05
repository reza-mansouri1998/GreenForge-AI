"""Advisory monitoring contracts: explicit missing/unevaluated/flagged states."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenforge_ai.training.evaluation import anomaly_summary
from greenforge_ai.training.train_anomaly import (
    apply_models, assign_periods, attach_forecast_residuals,
    fit_anomaly, prepare_monitor, review_episodes,
)


@pytest.fixture
def observations() -> pd.DataFrame:
    return pd.DataFrame({
        "WsDateTime": pd.date_range("2024-01-01", periods=240, freq="15min"),
        "power_kw": np.zeros(240, dtype=float),
        "coverage": np.ones(240, dtype=float),
    })


def test_quality_states_preserve_missing_readings(observations: pd.DataFrame) -> None:
    sample = observations.iloc[:12].copy()
    sample.loc[2, "power_kw"] = np.nan
    sample.loc[5, "coverage"] = 0.50
    sample.loc[7, "coverage"] = np.nan
    result = prepare_monitor(sample, "IPE_PV")
    assert result.loc[2, "quality_status"] == "missing_power"
    assert result.loc[5, "quality_status"] == "low_coverage"
    assert result.loc[7, "quality_status"] == "coverage_unknown"
    assert result.loc[5, "state"] == "not_observed"
    assert pd.isna(result.loc[5, "change_kw"])
    assert sample.loc[5, "power_kw"] == 0  # Original input was not mutated.


def test_invalid_monitor_timestamps_fail_early(observations: pd.DataFrame) -> None:
    wrong = observations.iloc[:10].drop(index=4).reset_index(drop=True)
    with pytest.raises(ValueError, match="complete 15-minute"):
        prepare_monitor(wrong, "TEC_48S")
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_monitor(observations, "UNKNOWN")


def test_chronological_periods_do_not_use_assessment_as_reference(observations: pd.DataFrame) -> None:
    prepared = prepare_monitor(observations, "TEC_48S")
    split = assign_periods(prepared)
    assert set(split["period"].unique()) == {"reference", "calibration", "assessment"}
    assert split.loc[split["period"].eq("reference"), "WsDateTime"].max() < (
        split.loc[split["period"].eq("calibration"), "WsDateTime"].min()
    )
    assert split.loc[split["period"].eq("calibration"), "WsDateTime"].max() < (
        split.loc[split["period"].eq("assessment"), "WsDateTime"].min()
    )


def test_insufficient_history_is_not_called_normal(observations: pd.DataFrame) -> None:
    short = observations.iloc[:80].copy()
    fitted = fit_anomaly("TEC_48S", short)
    monitor = fitted.monitoring
    assert fitted.models == {}
    assert fitted.summary["scored_assessment_intervals"] == 0
    assert fitted.summary["review_rate_pct"] is None
    assert not monitor["review_flag"].any()
    assert monitor["model_status"].eq("insufficient_reference").all()
    assert monitor["advisory"].eq("not_assessed").all()


def test_invalid_measurement_advisory_is_not_a_fault(observations: pd.DataFrame) -> None:
    frame = prepare_monitor(observations.iloc[:10], "TEC_48S")
    frame.loc[3, "quality_status"] = "missing_power"
    frame.loc[3, "state"] = "not_observed"
    assessed = apply_models(frame, "TEC_48S", models={})
    assert assessed.loc[3, "model_status"] == "data_quality"
    assert assessed.loc[3, "advisory"] == "measurement_unavailable"
    assert not bool(assessed.loc[3, "review_flag"])
    assert assessed.loc[0, "advisory"] == "not_assessed"


def test_summary_uses_only_scored_assessment_denominator() -> None:
    rows = pd.DataFrame({
        "period": ["reference", "assessment", "assessment", "assessment", "assessment"],
        "model_status": ["scored", "scored", "scored", "insufficient_reference", "data_quality"],
        "quality_status": ["valid", "valid", "valid", "valid", "low_coverage"],
        "review_flag": [False, True, False, False, False],
    })
    report = anomaly_summary(rows, "IPE_PV")
    assert report["intervals"] == 5
    assert report["quality_valid_intervals"] == 4
    assert report["scored_assessment_intervals"] == 2
    assert report["review_intervals"] == 1
    assert report["review_rate_pct"] == 50.0
    assert report["confirmed_faults"] is None


def test_review_episodes_join_only_adjacent_flags() -> None:
    times = pd.to_datetime([
        "2024-11-08 08:00", "2024-11-08 08:15", "2024-11-08 09:00",
        "2024-11-08 08:00", "2024-11-08 08:15",
    ])
    frame = pd.DataFrame({
        "WsDateTime": times,
        "system": ["TEC_48S"] * 3 + ["IPE_PV"] * 2,
        "power_kw": [0.1, 0.3, 0.4, 0.0, 0.0],
        "surprise_score": [0.9, 1.6, np.nan, 2.0, 1.5],
        "review_reason": ["Unusual pattern"] * 5,
        "review_flag": [True, True, True, True, False],
    })
    episodes = review_episodes(frame)
    assert len(episodes) == 3
    machine = episodes.loc[episodes["system"].eq("TEC_48S")]
    assert sorted(machine["intervals"].tolist()) == [1, 2]
    assert machine["end_exclusive"].max() == pd.Timestamp("2024-11-08 09:15")
    solar = episodes.loc[episodes["system"].eq("IPE_PV")]
    assert solar.iloc[0]["intervals"] == 1


def test_forecast_residuals_only_after_observed_and_available() -> None:
    times = pd.to_datetime(["2024-11-08 08:15", "2024-11-08 08:30", "2024-11-08 08:45"])
    monitor = pd.DataFrame({
        "WsDateTime": times,
        "power_kw": [2.0, 1.0, 3.0],
        "quality_status": ["valid", "valid", "low_coverage"],
        "interval_end": times + pd.Timedelta(minutes=15),
    })
    predictions = pd.DataFrame({
        "target_time": times,
        "prediction": [1.7, 0.7, 2.7],
        "available_at": [times[0], times[1] + pd.Timedelta(minutes=16), times[2]],
    })
    result = attach_forecast_residuals(monitor, predictions)
    assert len(result) == len(monitor)
    assert result.loc[0, "forecast_error_kw"] == pytest.approx(0.3)
    assert pd.isna(result.loc[1, "forecast_error_kw"])  # Too late.
    assert pd.isna(result.loc[2, "forecast_error_kw"])  # Unreliable reading.


def test_duplicate_forecast_handoff_rejected() -> None:
    base = pd.Timestamp("2024-11-08 08:15")
    monitor = pd.DataFrame({
        "WsDateTime": [base], "power_kw": [1.0],
        "quality_status": ["valid"],
        "interval_end": [base + pd.Timedelta(minutes=15)],
    })
    predictions = pd.DataFrame({
        "target_time": [base, base], "prediction": [0.4, 0.5],
        "available_at": [base, base],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        attach_forecast_residuals(monitor, predictions)
