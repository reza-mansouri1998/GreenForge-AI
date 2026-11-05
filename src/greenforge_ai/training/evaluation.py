"""Leak-safe forecasting comparisons and advisory anomaly-review statistics.

All comparisons use identical timestamps; missing persistence observations are
excluded from BOTH methods rather than silently giving one method an advantage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forecast_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    """R², MAE, MSE and RMSE (power in kW); never silently discard bad rows."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) < 2:
        raise ValueError("Metrics require at least two matched observations")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite actual and predicted values")
    mse = float(mean_squared_error(y, p))
    return {
        "r2": float(r2_score(y, p)) if np.var(y) > 0 else float("nan"),
        "mae_kw": float(mean_absolute_error(y, p)),
        "mse_kw2": mse,
        "rmse_kw": float(np.sqrt(mse)),
        "observations": int(len(y)),
    }


def persistence_metrics(test: pd.DataFrame) -> dict[str, float] | None:
    """Baseline: predict the next interval using the latest available reading."""
    if missing := {"value", "target"} - set(test):
        raise ValueError(f"Persistence baseline lacks {sorted(missing)}")
    eligible = test.loc[test["value"].notna(), ["value", "target"]]
    if len(eligible) < 2:
        return None
    return forecast_metrics(eligible["target"], eligible["value"])


def forecast_diagnostics(
    predictions: pd.DataFrame, *, active_threshold_kw: float,
) -> dict[str, float | int | bool | None]:
    """Compare ML and persistence on the SAME rows, including active intervals.

    ``prediction`` is the deployable, postprocessed forecast. If present,
    ``raw_prediction`` records the original estimator output for auditing.
    The active threshold is a reporting convention, not an equipment limit.
    """
    required = {"target", "prediction", "value"}
    if missing := required - set(predictions):
        raise ValueError(f"Forecast diagnostics lack {sorted(missing)}")
    if not np.isfinite(active_threshold_kw) or active_threshold_kw <= 0:
        raise ValueError("active_threshold_kw must be positive and finite")
    numeric = predictions.loc[:, ["target", "prediction", "value"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric["target"]).all() or not np.isfinite(numeric["prediction"]).all():
        raise ValueError("Forecast target and model prediction must be finite")
    present = numeric["value"].notna()
    if not np.isfinite(numeric.loc[present, "value"]).all():
        raise ValueError("Observed persistence values must be finite")
    comparable = numeric.loc[present]
    matched = len(comparable)
    result: dict[str, float | int | bool | None] = {
        "matched_observations": matched,
        "unavailable_persistence_observations": int((~present).sum()),
        "active_threshold_kw": float(active_threshold_kw),
        "active_matched_observations": int((comparable["target"] >= active_threshold_kw).sum()),
        "test_near_zero_target_share": float((numeric["target"].abs() < 1e-8).mean()),
        "model_mae_on_matched_kw": None,
        "persistence_mae_on_matched_kw": None,
        "model_rmse_on_matched_kw": None,
        "persistence_rmse_on_matched_kw": None,
        "model_beats_persistence_mae": None,
        "active_model_mae_kw": None,
        "active_persistence_mae_kw": None,
    }
    if "raw_prediction" in predictions:
        raw = pd.to_numeric(predictions["raw_prediction"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Raw forecast predictions must be finite")
        result["raw_negative_predictions"] = int((raw < 0).sum())
        result["postprocessed_predictions"] = int((raw != numeric["prediction"].to_numpy()).sum())
    if matched >= 2:
        ml = forecast_metrics(comparable["target"], comparable["prediction"])
        base = forecast_metrics(comparable["target"], comparable["value"])
        result.update({
            "model_mae_on_matched_kw": ml["mae_kw"],
            "persistence_mae_on_matched_kw": base["mae_kw"],
            "model_rmse_on_matched_kw": ml["rmse_kw"],
            "persistence_rmse_on_matched_kw": base["rmse_kw"],
            "model_beats_persistence_mae": bool(ml["mae_kw"] < base["mae_kw"]),
        })
        active = comparable.loc[comparable["target"] >= active_threshold_kw]
        if len(active) >= 2:
            result["active_model_mae_kw"] = forecast_metrics(active["target"], active["prediction"])["mae_kw"]
            result["active_persistence_mae_kw"] = forecast_metrics(active["target"], active["value"])["mae_kw"]
    return result


def anomaly_summary(monitor: pd.DataFrame, system: str) -> dict[str, object]:
    """Report review workload, never guessed fault-detection accuracy."""
    required = {"period", "model_status", "quality_status", "review_flag"}
    if missing := required - set(monitor):
        raise ValueError(f"Anomaly report lacks {sorted(missing)}")
    scored = monitor.loc[
        monitor["period"].eq("assessment") & monitor["model_status"].eq("scored")
    ]
    flags = int(scored["review_flag"].sum())
    return {
        "system": system,
        "intervals": int(len(monitor)),
        "quality_valid_intervals": int(monitor["quality_status"].eq("valid").sum()),
        "scored_assessment_intervals": int(len(scored)),
        "review_intervals": flags,
        "review_rate_pct": round(100 * flags / len(scored), 4) if len(scored) else None,
        "confirmed_faults": None,
    }
