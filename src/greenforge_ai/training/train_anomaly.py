"""Notebook 02's regime-aware, human-review anomaly monitoring.

No fault labels exist: review flags are advisory, not fault diagnoses.
Persist fitted estimators, imputers, scalers and calibration thresholds together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from greenforge_ai.training.evaluation import anomaly_summary
from greenforge_ai.training.validation import INTERVAL

SEED = 42
MIN_COVERAGE = 0.80
REVIEW_SHARE = 0.02
MIN_TRAIN, MIN_CALIB = 40, 20
MAX_REFERENCE = 3000
MACHINE_STEP_REVIEW_KW = 0.40
IDLE_EPS_KW = 1e-6
INGESTION_DELAY_MINUTES = 0  # Historical assumption; not a measured live latency.
SYSTEM_BANDS = {"TEC_48S": 0.04, "IPE_PV": 0.20}
# The exact notebook-selected model families and settings, not a new search.
SELECTED = {
    "TEC_48S": {"lower_load": ("PCA", 2), "higher_load": ("LOF", 40)},
    "IPE_PV": {"lower_load": ("PCA", 2), "higher_load": ("KMeans", 8)},
}
FEATURES = (
    "power_kw", "previous_kw", "change_kw", "past_mean_4", "past_std_8",
    "hour_sin", "hour_cos",
)


@dataclass
class BandModel:
    method: str
    setting: int
    features: tuple[str, ...]
    preparation: object
    detector: object
    center: float
    spread: float
    raw_threshold: float
    reference_rows: int
    calibration_rows: int


@dataclass
class AnomalyResult:
    system: str
    monitoring: pd.DataFrame
    models: dict[str, BandModel]
    summary: dict[str, object]
    trained_before: str | None


def prepare_monitor(observations: pd.DataFrame, system: str) -> pd.DataFrame:
    """Build quality-aware features without filling real measurement gaps."""
    if system not in SELECTED:
        raise ValueError(f"Unsupported monitored system: {system}")
    needed = {"WsDateTime", "power_kw", "coverage"}
    if needed - set(observations):
        raise ValueError(f"Missing anomaly inputs: {sorted(needed - set(observations))}")
    frame = observations.loc[:, ["WsDateTime", "power_kw", "coverage"]].copy().reset_index(drop=True)
    if frame["WsDateTime"].duplicated().any() or not frame["WsDateTime"].is_monotonic_increasing:
        raise ValueError("Anomaly measurements must be unique and sorted")
    if not frame["WsDateTime"].diff().dropna().eq(INTERVAL).all():
        raise ValueError("Anomaly measurements must use a complete 15-minute grid")
    frame["quality_status"] = np.select(
        [frame["power_kw"].isna(), frame["coverage"].isna(), frame["coverage"] < MIN_COVERAGE],
        ["missing_power", "coverage_unknown", "low_coverage"], default="valid",
    )
    power = frame["power_kw"].where(frame["quality_status"].eq("valid"))
    frame["previous_kw"] = power.shift(1)
    frame["change_kw"] = power - frame["previous_kw"]
    frame["past_mean_4"] = power.shift(1).rolling(4, min_periods=2).mean()
    frame["past_std_8"] = power.shift(1).rolling(8, min_periods=4).std()
    prior = frame["past_mean_4"].fillna(frame["previous_kw"]).fillna(power)
    frame["state"] = np.where(prior >= SYSTEM_BANDS[system], "higher_load", "lower_load")
    frame.loc[frame["quality_status"].ne("valid"), "state"] = "not_observed"
    time = frame["WsDateTime"]
    hour = (time.dt.hour + time.dt.minute / 60) / 24
    frame["hour_sin"] = np.sin(2 * np.pi * hour)
    frame["hour_cos"] = np.cos(2 * np.pi * hour)
    frame["period"] = "no_observation"
    return frame


def assign_periods(
    frame: pd.DataFrame, *,
    early_cutoff: pd.Timestamp | None = None,
    early_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reference/calibration/assessment splits use time-ordered, valid data."""
    out = frame.copy()
    usable = out["quality_status"].eq("valid")
    good = np.flatnonzero(usable.to_numpy())
    if early_cutoff is None:
        if len(good) < 200:
            out.loc[usable, "period"] = "insufficient_history"
            return out
        a, b = good[int(len(good) * 0.60)], good[int(len(good) * 0.80)]
        out.loc[good[good < a], "period"] = "reference"
        out.loc[good[(good >= a) & (good < b)], "period"] = "calibration"
        out.loc[good[good >= b], "period"] = "assessment"
    else:
        if early_end is None or early_end <= early_cutoff:
            raise ValueError("Early assessment requires ordered cutoff and end")
        stamps = out["WsDateTime"]
        before = good[stamps.iloc[good].lt(early_cutoff).to_numpy()]
        if len(before) < 200:
            raise ValueError("Insufficient quality-valid history before the early monitoring cutoff")
        cut = int(0.70 * len(before))
        out.loc[before[:cut], "period"] = "reference"
        out.loc[before[cut:], "period"] = "calibration"
        assess = usable & stamps.ge(early_cutoff) & stamps.lt(early_end)
        out.loc[assess, "period"] = "assessment"
    return out


def _fit_detector(method: str, setting: int, x: np.ndarray) -> object:
    if method == "PCA":
        detector = PCA(n_components=min(setting, x.shape[1], len(x)), svd_solver="full")
    elif method == "LOF":
        detector = LocalOutlierFactor(n_neighbors=min(setting, len(x) - 1), novelty=True, n_jobs=2)
    elif method == "KMeans":
        detector = MiniBatchKMeans(n_clusters=min(setting, len(x)), batch_size=512, n_init=3, random_state=SEED)
    elif method == "Isolation Forest":
        detector = IsolationForest(n_estimators=96, max_samples=min(128, len(x)), n_jobs=2, random_state=SEED)
    else:
        raise ValueError(f"Unknown anomaly detector: {method}")
    if method == "KMeans":
        with threadpool_limits(limits=1, user_api="openmp"):
            return detector.fit(x)
    return detector.fit(x)


def _raw_scores(method: str, detector: object, x: np.ndarray) -> np.ndarray:
    if method in ("LOF", "Isolation Forest"):
        return -detector.score_samples(x)
    if method == "KMeans":
        return np.square(detector.transform(x)).min(axis=1)
    if method == "PCA":
        reconstructed = detector.inverse_transform(detector.transform(x))
        return np.square(x - reconstructed).mean(axis=1)
    raise ValueError(f"Unknown anomaly detector: {method}")


def _pv_episode_filter(frame: pd.DataFrame, flags: pd.Series) -> pd.Series:
    """Show the first unusual steady-zero solar reading in each daily episode."""
    zeros = (
        frame["quality_status"].eq("valid")
        & frame["power_kw"].abs().le(IDLE_EPS_KW)
        & frame["previous_kw"].abs().le(IDLE_EPS_KW)
    )
    day = frame["WsDateTime"].dt.normalize()
    episode = (
        zeros.ne(zeros.shift(fill_value=False))
        | frame["WsDateTime"].diff().ne(INTERVAL)
        | day.ne(day.shift())
    ).cumsum()
    repeated = (flags & zeros) & (flags & zeros).groupby(episode, sort=False).cumsum().gt(1)
    return flags & ~repeated


def apply_models(frame: pd.DataFrame, system: str, models: dict[str, BandModel]) -> pd.DataFrame:
    """Score with fitted reference/calibration models; never learn on assessment."""
    out = frame.copy()
    out["surprise_score"] = np.nan
    out["model_flag"] = False
    out["transition_flag"] = False
    out["review_flag"] = False
    out["model_status"] = np.where(out["quality_status"].eq("valid"), "insufficient_reference", "data_quality")
    stable_idle = (
        out["power_kw"].abs().le(IDLE_EPS_KW)
        & out["previous_kw"].abs().le(IDLE_EPS_KW)
    ) if system == "TEC_48S" else pd.Series(False, index=out.index)

    for state, fitted in models.items():
        eligible = out["quality_status"].eq("valid") & out["state"].eq(state)
        if not eligible.any():
            continue
        x = fitted.preparation.transform(out.loc[eligible, list(fitted.features)])
        raw = _raw_scores(fitted.method, fitted.detector, x)
        out.loc[eligible, "surprise_score"] = (raw - fitted.center) / fitted.spread
        out.loc[eligible, "model_status"] = "scored"
        assessed = eligible & out["period"].eq("assessment")
        alert = assessed & out["surprise_score"].gt(1.0) & ~stable_idle
        if system == "IPE_PV":
            alert = _pv_episode_filter(out, alert)
        transition = (
            assessed & out["change_kw"].abs().ge(MACHINE_STEP_REVIEW_KW)
        ) if system == "TEC_48S" else pd.Series(False, index=out.index)
        out.loc[alert, "model_flag"] = True
        out.loc[transition, "transition_flag"] = True
        out.loc[alert | transition, "review_flag"] = True

    out["review_reason"] = np.select(
        [out["model_flag"] & out["transition_flag"], out["transition_flag"], out["model_flag"]],
        ["Pattern and large change", "Large power change", "Unusual pattern"],
        default="none",
    )
    out["review_type"] = np.select(
        [out["quality_status"].ne("valid"), out["review_flag"]],
        ["data_quality", "unusual_observation"], default="none",
    )
    out["interval_end"] = out["WsDateTime"] + INTERVAL
    out["available_at"] = out["interval_end"] + pd.Timedelta(minutes=INGESTION_DELAY_MINUTES)
    out["advisory"] = np.select(
        [out["quality_status"].ne("valid"),
         out["period"].ne("assessment") | out["model_status"].ne("scored"),
         out["review_flag"]],
        ["measurement_unavailable", "not_assessed", "review_observation"],
        default="no_model_alert",
    )
    out.insert(1, "system", system)
    return out


def fit_anomaly(
    system: str, observations: pd.DataFrame, *,
    early_cutoff: pd.Timestamp | None = None,
    early_end: pd.Timestamp | None = None,
) -> AnomalyResult:
    """Fit only on historical reference data; calibrate only on the next period."""
    data = assign_periods(prepare_monitor(observations, system), early_cutoff=early_cutoff, early_end=early_end)
    models: dict[str, BandModel] = {}
    for state, (method, setting) in SELECTED[system].items():
        ref = data.loc[data["period"].eq("reference") & data["state"].eq(state)]
        cal = data.loc[data["period"].eq("calibration") & data["state"].eq(state)]
        if len(ref) < MIN_TRAIN or len(cal) < MIN_CALIB:
            continue
        features = tuple(name for name in FEATURES if ref[name].notna().mean() >= 0.5)
        if len(features) < 2:
            continue
        sample = ref.iloc[np.linspace(0, len(ref) - 1, min(len(ref), MAX_REFERENCE), dtype=int)]
        prep = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
        x = prep.fit_transform(sample.loc[:, features])
        detector = _fit_detector(method, setting, x)
        cal_scores = _raw_scores(method, detector, prep.transform(cal.loc[:, features]))
        if system == "TEC_48S":
            quiet = cal["power_kw"].abs().le(IDLE_EPS_KW) & cal["previous_kw"].abs().le(IDLE_EPS_KW)
            actionable = cal_scores[~quiet.to_numpy()]
            if len(actionable) < MIN_CALIB:
                actionable = cal_scores
        else:
            actionable = cal_scores
        center = float(np.median(actionable))
        threshold = float(np.quantile(actionable, 1 - REVIEW_SHARE))
        spread = max(threshold - center, 1e-8)
        models[state] = BandModel(
            method, setting, features, prep, detector, center, spread, threshold,
            len(ref), len(cal),
        )
    output = apply_models(data, system, models)
    if early_cutoff is not None:
        output = output.loc[
            output["WsDateTime"].ge(early_cutoff) & output["WsDateTime"].lt(early_end)
        ].reset_index(drop=True)
    return AnomalyResult(
        system, output, models, anomaly_summary(output, system),
        str(early_cutoff) if early_cutoff is not None else None,
    )


def attach_forecast_residuals(
    monitoring: pd.DataFrame, predictions: pd.DataFrame | None
) -> pd.DataFrame:
    """Retrospective forecast confirmation, only after observations are available.

    Never use these after-the-fact errors to train detectors or influence MILP.
    """
    result = monitoring.copy()
    result["forecast_kw"] = np.nan
    result["forecast_error_kw"] = np.nan
    if predictions is None or predictions.empty:
        return result
    needed = {"target_time", "prediction", "available_at"}
    if missing := needed - set(predictions):
        raise ValueError(f"Forecast monitoring handoff lacks {sorted(missing)}")
    if predictions["target_time"].duplicated().any():
        raise ValueError("Duplicate forecast timestamps in monitoring handoff")
    lookup = predictions.loc[:, ["target_time", "prediction", "available_at"]].rename(
        columns={"target_time": "WsDateTime", "prediction": "_prediction",
                 "available_at": "_forecast_issued_at"}
    )
    merged = result.merge(lookup, on="WsDateTime", how="left", validate="one_to_one", sort=False)
    if len(merged) != len(result):
        raise AssertionError("Forecast handoff changed monitoring row count")
    eligible = (
        merged["_prediction"].notna()
        & merged["quality_status"].eq("valid")
        & merged["_forecast_issued_at"].le(merged["interval_end"])
    )
    merged.loc[eligible, "forecast_kw"] = merged.loc[eligible, "_prediction"]
    merged.loc[eligible, "forecast_error_kw"] = (
        merged.loc[eligible, "power_kw"] - merged.loc[eligible, "_prediction"]
    )
    return merged.drop(columns=["_prediction", "_forecast_issued_at"])


def review_episodes(monitoring: pd.DataFrame) -> pd.DataFrame:
    """Bundle adjacent reviewed 15-minute intervals into readable events."""
    flagged = monitoring.loc[monitoring["review_flag"],
                             ["WsDateTime", "system", "power_kw", "surprise_score", "review_reason"]]
    columns = ["system", "start", "end_exclusive", "intervals", "max_surprise_score", "review_reasons"]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for system, group in flagged.groupby("system", sort=True):
        ordered = group.sort_values("WsDateTime").copy()
        segment = ordered["WsDateTime"].diff().ne(INTERVAL).cumsum()
        for _, event in ordered.groupby(segment):
            records.append({
                "system": system,
                "start": event["WsDateTime"].iloc[0],
                "end_exclusive": event["WsDateTime"].iloc[-1] + INTERVAL,
                "intervals": len(event),
                "max_surprise_score": (float(event["surprise_score"].max())
                                       if event["surprise_score"].notna().any() else None),
                "review_reasons": "; ".join(sorted(set(event["review_reason"]))),
            })
    return pd.DataFrame.from_records(records, columns=columns)
