"""Train leak-safe, next-15-minute TEC/PV forecasts from ETL EDA observations.

Notebook 01 is the reference: its feature recipe and tuned settings are used,
not its exploratory test-R² champion-selection procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from greenforge_ai.training.evaluation import (
    forecast_diagnostics, forecast_metrics, persistence_metrics,
)
from greenforge_ai.training.validation import (
    FORECAST_FEATURES, INTERVAL, chronological_train_test, verify_predictions,
)

SEED = 42


@dataclass
class ForecastResult:
    system: str
    model_name: str
    model: Pipeline
    predictions: pd.DataFrame
    test_metrics: dict[str, float]
    baseline_metrics: dict[str, float] | None
    selection_scores: dict[str, float]
    train_rows: int
    test_rows: int
    model_parameters: dict[str, object]
    comparison_metrics: dict[str, float | int | bool | None]
    postprocessing: str
    input_example: pd.DataFrame


def make_forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Notebook 01's 30 features; usable for fitting AND future inference.

    WsDateTime = start of last measured 15-minute interval. Its value is known
    at WsDateTime + 15 min (an assumed zero-ingestion-delay issue time).
    Only the *label*, never a model feature, is shifted into the future.
    """
    if {"WsDateTime", "power_kw"} - set(frame):
        raise ValueError("Forecast features need WsDateTime and power_kw")
    if not frame["WsDateTime"].is_monotonic_increasing:
        raise ValueError("Forecast input must be ordered by timestamp")
    if frame["WsDateTime"].duplicated().any():
        raise ValueError("Duplicate forecast observation timestamps")
    d = frame.loc[:, ["WsDateTime"]].copy()
    y = pd.to_numeric(frame["power_kw"], errors="raise").astype(float)
    d["value"] = y.to_numpy()
    for k in (1, 2, 4, 8, 12, 24, 48, 96, 672):
        d[f"lag{k}"] = y.shift(k).to_numpy()
    past = y.shift(1)
    for k in (4, 8, 24, 96):
        d[f"roll_mean_{k}"] = past.rolling(k, min_periods=max(2, k // 4)).mean().to_numpy()
    for k in (8, 24, 96):
        d[f"roll_std_{k}"] = past.rolling(k, min_periods=max(2, k // 4)).std().to_numpy()
    d["roll_min_24"] = past.rolling(24, min_periods=6).min().to_numpy()
    d["roll_max_24"] = past.rolling(24, min_periods=6).max().to_numpy()
    d["missing_24"] = past.isna().rolling(24, min_periods=6).mean().to_numpy()
    d["active_24"] = (past.fillna(0).abs() > 0.1).rolling(24, min_periods=6).mean().to_numpy()
    d["change_1"] = (y.shift(1) - y.shift(2)).to_numpy()
    d["change_4"] = (y.shift(1) - y.shift(5)).to_numpy()
    t = d["WsDateTime"]
    phases = {
        "hour": (t.dt.hour + t.dt.minute / 60) / 24,
        "dow": t.dt.dayofweek / 7,
        "month": (t.dt.month - 1) / 12,
    }
    for name, phase in phases.items():
        d[f"{name}_sin"] = np.sin(2 * np.pi * phase)
        d[f"{name}_cos"] = np.cos(2 * np.pi * phase)
    d["is_weekend"] = (t.dt.dayofweek >= 5).astype(np.int8)
    d["target"] = y.shift(-1).to_numpy()
    d["target_time"] = t + INTERVAL
    if set(FORECAST_FEATURES) - set(d):
        raise AssertionError("Forecast recipe and feature contract are inconsistent")
    return d.reset_index(drop=True)


def estimator_for(name: str) -> Pipeline:
    """Small, fixed notebook-derived models. No expensive parameter search."""
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(
            n_estimators=180, num_leaves=7, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=2, verbosity=-1,
            random_state=SEED,
        )
    elif name == "CatBoost":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(
            iterations=180, depth=3, learning_rate=0.05,
            thread_count=2, verbose=False, random_seed=SEED,
        )
    elif name == "Ridge":
        estimator = Ridge(alpha=20.0)
    else:
        raise ValueError(f"Unknown notebook-derived model: {name}")
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name == "Ridge":
        steps.append(("scaler", StandardScaler()))
    return Pipeline([*steps, ("model", estimator)])


def train_forecast(
    system: str,
    observations: pd.DataFrame,
    *,
    select_by_cv: bool = False,
) -> ForecastResult:
    """Use CV-only champions by default; optionally compare TWO fixed candidates.

    'Notebook-CV' preset: TEC=LightGBM (7 leaves), PV=Ridge (alpha 20).
    Opt-in candidate comparison uses TimeSeriesSplit on training data only.
    The final 20% test partition never chooses the estimator.
    """
    presets = {
        "TEC_48S": ("LightGBM", "CatBoost"),
        "IPE_PV": ("Ridge", "CatBoost"),
    }
    if system not in presets:
        raise ValueError(f"Unknown system: {system}")
    full = make_forecast_features(observations)
    supervised = full.loc[full["target"].notna()].reset_index(drop=True)
    train, test = chronological_train_test(supervised, test_share=0.20, gap_intervals=4)
    x_train = train.loc[:, FORECAST_FEATURES]
    y_train = train["target"]
    candidates = presets[system] if select_by_cv else presets[system][:1]
    scores: dict[str, float] = {}
    if select_by_cv:
        cv = TimeSeriesSplit(n_splits=3, gap=4, max_train_size=16_000)
        for candidate in candidates:
            score = cross_val_score(
                estimator_for(candidate), x_train, y_train, cv=cv,
                scoring="r2", n_jobs=1, error_score="raise",
            )
            if not np.isfinite(score).all():
                raise ValueError(f"Non-finite cross-validation result: {candidate}")
            scores[candidate] = float(score.mean())
        selected = max(scores, key=scores.get)
    else:
        selected = candidates[0]
    model = estimator_for(selected)
    model.fit(x_train, y_train)
    prediction = model.predict(test.loc[:, FORECAST_FEATURES])
    results = test.loc[:, ["WsDateTime", "target_time", "target", "value"]].copy()
    raw_predictions = np.asarray(prediction, dtype=float)
    results["raw_prediction"] = raw_predictions
    # Solar generation cannot be negative. Preserve raw outputs for auditing,
    # but use nonnegative solar forecasts for evaluation and scheduling.
    # TEC_48S remains signed because active electrical power can be bidirectional;
    # a nonnegative machine-demand assumption needs separate validation.
    results["prediction"] = (
        np.maximum(raw_predictions, 0.0) if system == "IPE_PV" else raw_predictions
    )
    # Interval's measurement ends at its target_time; availability is assumed
    # immediate. Actual acquisition delay must be configured for live serving.
    results["available_at"] = results["target_time"]
    verify_predictions(results)
    test_metrics = forecast_metrics(results["target"], results["prediction"])
    comparison = forecast_diagnostics(
        results, active_threshold_kw=0.1 if system == "TEC_48S" else 0.2,
    )
    return ForecastResult(
        system=system,
        model_name=selected,
        model=model,
        predictions=results.reset_index(drop=True),
        test_metrics=test_metrics,
        baseline_metrics=persistence_metrics(test),
        selection_scores=scores,
        train_rows=len(train),
        test_rows=len(test),
        model_parameters={key: value for key, value in model.named_steps["model"].get_params().items()
                          if key in {"alpha", "n_estimators", "iterations", "num_leaves", "depth",
                                     "learning_rate", "random_state", "random_seed"}},
        comparison_metrics=comparison,
        postprocessing="nonnegative_pv_only" if system == "IPE_PV" else "none_signed_tec_power",
        input_example=test.loc[:, FORECAST_FEATURES].tail(3).astype("float64").copy(),
    )


def predict_next(
    model: Pipeline, observations: pd.DataFrame, *, system: str | None = None,
) -> dict[str, object]:
    """Make a genuinely new 15-minute forecast without a future measurement."""
    features = make_forecast_features(observations)
    current = features.iloc[-1:]
    if pd.isna(current["value"].iloc[0]):
        raise ValueError("Current interval has no measured power; cannot issue a forecast")
    prediction = float(model.predict(current.loc[:, FORECAST_FEATURES])[0])
    if not np.isfinite(prediction):
        raise ValueError("Forecast is non-finite")
    if system is not None and system not in {"TEC_48S", "IPE_PV"}:
        raise ValueError(f"Unknown forecast system: {system}")
    if system == "IPE_PV":
        prediction = max(prediction, 0.0)
    return {
        "WsDateTime": current["WsDateTime"].iloc[0],
        "target_time": current["target_time"].iloc[0],
        "available_at": current["target_time"].iloc[0],
        "prediction_kw": prediction,
    }
