"""Fast, leak-safe 15-minute forecasting unit tests; no full model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenforge_ai.training import train_forecasting as training
from greenforge_ai.training.evaluation import (
    forecast_diagnostics, forecast_metrics, persistence_metrics,
)
from greenforge_ai.training.validation import (
    FORECAST_FEATURES, chronological_train_test, verify_predictions,
)


@pytest.fixture
def history() -> pd.DataFrame:
    n = 760  # Only 760 points; enough for the 672-interval lookback.
    return pd.DataFrame({
        "WsDateTime": pd.date_range("2024-10-01", periods=n, freq="15min"),
        "power_kw": np.arange(n, dtype=float) / 100.0,
    })


def test_lags_rolling_features_and_target_cannot_look_ahead(history: pd.DataFrame) -> None:
    features = training.make_forecast_features(history)
    assert len(FORECAST_FEATURES) == 30
    assert set(FORECAST_FEATURES).issubset(features)
    assert features.loc[672, "lag672"] == pytest.approx(history.loc[0, "power_kw"])
    assert features.loc[672, "lag1"] == pytest.approx(history.loc[671, "power_kw"])
    assert features.loc[672, "roll_mean_4"] == pytest.approx(
        history.loc[668:671, "power_kw"].mean()
    )
    assert features.loc[672, "target"] == pytest.approx(history.loc[673, "power_kw"])
    assert features.loc[672, "target_time"] == history.loc[672, "WsDateTime"] + pd.Timedelta(minutes=15)
    assert np.isnan(features.iloc[-1]["target"])


@pytest.mark.parametrize("corruption", ["reverse", "duplicate", "missing_column"])
def test_feature_builder_rejects_bad_input(history: pd.DataFrame, corruption: str) -> None:
    if corruption == "reverse":
        history = history.iloc[::-1].reset_index(drop=True)
    elif corruption == "duplicate":
        history.loc[10, "WsDateTime"] = history.loc[9, "WsDateTime"]
    else:
        history = history.drop(columns="power_kw")
    with pytest.raises(ValueError):
        training.make_forecast_features(history)


def test_chronological_split_enforces_embargo(history: pd.DataFrame) -> None:
    supervised = training.make_forecast_features(history).dropna(subset=["target"])
    train, test = chronological_train_test(supervised, gap_intervals=4)
    assert train["WsDateTime"].max() < test["WsDateTime"].min()
    assert train["target_time"].max() < test["WsDateTime"].min()
    assert len(supervised) == len(train) + len(test) + 4


def test_forecast_metric_values_and_invalid_inputs() -> None:
    scores = forecast_metrics([0.0, 2.0], [0.0, 1.0])
    assert scores["mae_kw"] == pytest.approx(0.5)
    assert scores["mse_kw2"] == pytest.approx(0.5)
    assert scores["rmse_kw"] == pytest.approx(np.sqrt(0.5))
    with pytest.raises(ValueError, match="finite"):
        forecast_metrics([0.0, np.nan], [0.0, 1.0])
    with pytest.raises(ValueError):
        forecast_metrics([1.0], [1.0])


def test_fair_persistence_comparison_uses_identical_rows() -> None:
    rows = pd.DataFrame({
        "target": [1.0, 1.0, 0.0, 2.0],
        "value": [0.0, np.nan, 0.0, 1.0],
        "raw_prediction": [-0.2, 1.0, 0.2, 1.8],
        "prediction": [0.0, 1.0, 0.2, 1.8],
    })
    result = forecast_diagnostics(rows, active_threshold_kw=0.5)
    assert result["matched_observations"] == 3
    assert result["unavailable_persistence_observations"] == 1
    assert result["active_matched_observations"] == 2
    assert result["raw_negative_predictions"] == 1
    assert result["postprocessed_predictions"] == 1
    assert result["model_mae_on_matched_kw"] == pytest.approx((1 + 0.2 + 0.2) / 3)
    assert result["persistence_mae_on_matched_kw"] == pytest.approx(2 / 3)
    assert persistence_metrics(rows)["mae_kw"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("wrong_target", ["shift", "duplicate"])
def test_prediction_contract_rejects_wrong_horizon(wrong_target: str) -> None:
    times = pd.date_range("2024-01-01", periods=2, freq="15min")
    table = pd.DataFrame({
        "WsDateTime": times,
        "target_time": times + pd.Timedelta(minutes=15),
        "prediction": [1.0, 2.0],
    })
    if wrong_target == "shift":
        table.loc[1, "target_time"] += pd.Timedelta(minutes=15)
    else:
        table.loc[1, "target_time"] = table.loc[0, "target_time"]
    with pytest.raises(ValueError):
        verify_predictions(table)


class _FakeModel:
    def __init__(self, kind: str = "Ridge", value: float = -0.25) -> None:
        self.kind, self.value = kind, value
        self.named_steps = {"model": self}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> _FakeModel:
        assert x.index.equals(y.index)
        self.training_rows = len(x)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        assert list(x.columns) == list(FORECAST_FEATURES)
        return np.full(len(x), self.value, dtype=float)

    def get_params(self) -> dict:
        return {}


def test_next_forecast_clips_pv_only_and_uses_last_interval(history: pd.DataFrame) -> None:
    pv = training.predict_next(_FakeModel(value=-0.25), history, system="IPE_PV")
    tec = training.predict_next(_FakeModel(value=-0.25), history, system="TEC_48S")
    assert pv["prediction_kw"] == 0.0
    assert tec["prediction_kw"] == pytest.approx(-0.25)
    assert pv["target_time"] == history["WsDateTime"].iloc[-1] + pd.Timedelta(minutes=15)
    bad = history.copy()
    bad.loc[len(bad) - 1, "power_kw"] = np.nan
    with pytest.raises(ValueError, match="Current interval"):
        training.predict_next(_FakeModel(), bad, system="IPE_PV")


def test_cv_selection_never_uses_test_set_to_pick_winner(
    history: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_train_sizes = []

    def fake_cv(model: _FakeModel, x: pd.DataFrame, y: pd.Series, **kwargs) -> np.ndarray:
        seen_train_sizes.append(len(x))
        assert kwargs["scoring"] == "r2" and kwargs["n_jobs"] == 1
        return np.array([0.75] * 3 if model.kind == "CatBoost" else [0.70] * 3)

    monkeypatch.setattr(training, "estimator_for", lambda kind: _FakeModel(kind, 0.1))
    monkeypatch.setattr(training, "cross_val_score", fake_cv)
    result = training.train_forecast("TEC_48S", history, select_by_cv=True)
    assert result.model_name == "CatBoost"
    assert len(seen_train_sizes) == 2
    assert all(size == result.train_rows for size in seen_train_sizes)
    assert result.train_rows + result.test_rows < len(history)
    assert result.predictions["prediction"].eq(0.1).all()
    assert result.test_metrics["observations"] == result.test_rows


def test_preset_selection_avoids_unnecessary_cv(history: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "estimator_for", lambda name: _FakeModel(name, -0.1))
    monkeypatch.setattr(training, "cross_val_score", lambda *args, **kwargs: pytest.fail("CV not requested"))
    result = training.train_forecast("IPE_PV", history, select_by_cv=False)
    assert result.model_name == "Ridge"
    assert result.selection_scores == {}
    assert result.postprocessing == "nonnegative_pv_only"
    assert result.predictions["prediction"].eq(0).all()
    assert result.predictions["raw_prediction"].eq(-0.1).all()
