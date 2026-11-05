"""Read published 15-minute observations and enforce time-safe ML contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from greenforge_ai.features.interaction_features import artifact, manifest, project_root

INTERVAL = pd.Timedelta(minutes=15)
SYSTEMS = ("TEC_48S", "IPE_PV")
FORECAST_FEATURES = (
    "value", "lag1", "lag2", "lag4", "lag8", "lag12", "lag24", "lag48",
    "lag96", "lag672", "roll_mean_4", "roll_mean_8", "roll_mean_24",
    "roll_mean_96", "roll_std_8", "roll_std_24", "roll_std_96",
    "roll_min_24", "roll_max_24", "missing_24", "active_24",
    "change_1", "change_4", "hour_sin", "hour_cos", "dow_sin",
    "dow_cos", "month_sin", "month_cos", "is_weekend",
)


def require_published_target(root: Path, system: str) -> None:
    """The ETL may publish EDA data without approving its target for ML."""
    if system not in SYSTEMS:
        raise ValueError(f"Unsupported system: {system}")
    _, published = manifest(root)
    quality = published.get("reports", {}).get(system, {})
    features = quality.get("features", {})
    if not quality.get("target_quality_passed") or features.get("ml_status") != "ready":
        raise ValueError(f"ETL target is not ML-ready for {system}: {features}")
    config = published.get("config", {})
    if config.get("horizon_intervals") != 1:
        raise ValueError("This training pipeline supports only the verified 15-minute horizon")
    if config.get("timestamp_mode") != "publisher_wall_clock":
        raise ValueError("The notebook-derived pipeline requires publisher_wall_clock timestamps")


def validate_intervals(frame: pd.DataFrame, *, system: str) -> pd.DataFrame:
    """Check publisher clock labels without pretending naive times are UTC instants."""
    required = {"WsDateTime", "power_kw", "coverage"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{system}: missing columns {sorted(missing)}")
    result = frame.loc[:, ["WsDateTime", "power_kw", "coverage"]].copy()
    if not pd.api.types.is_datetime64_any_dtype(result["WsDateTime"]):
        raise TypeError(f"{system}: WsDateTime must be a Parquet datetime")
    if result["WsDateTime"].isna().any():
        raise ValueError(f"{system}: null timestamps")
    if not result["WsDateTime"].is_monotonic_increasing or result["WsDateTime"].duplicated().any():
        raise ValueError(f"{system}: timestamps must be unique and sorted")
    if not result["WsDateTime"].diff().dropna().eq(INTERVAL).all():
        raise ValueError(f"{system}: timestamps are not on a complete 15-minute grid")
    for column in ("power_kw", "coverage"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
        values = result[column].dropna().to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{system}: non-finite {column}")
    observed = result["coverage"].dropna()
    if not observed.between(0, 1).all():
        raise ValueError(f"{system}: coverage must be a fraction from 0 to 1")
    if result.empty:
        raise ValueError(f"{system}: empty time series")
    return result.reset_index(drop=True)


def load_eda(root: Path | str, system: str) -> pd.DataFrame:
    """Load only the small, ETL-verified 15-minute table; never read raw data."""
    root = project_root(root)
    require_published_target(root, system)
    path = artifact(system, "eda_15min.parquet", root)
    return validate_intervals(pd.read_parquet(path, columns=["WsDateTime", "power_kw", "coverage"]), system=system)


def chronological_train_test(
    supervised: pd.DataFrame, *, test_share: float = 0.20, gap_intervals: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological notebook-style 80/20 split with a one-hour embargo."""
    if not 0 < test_share < 0.5 or gap_intervals < 1:
        raise ValueError("Invalid chronological split configuration")
    expected = {"WsDateTime", "target_time", "target"}
    if expected - set(supervised):
        raise ValueError(f"Supervised frame lacks {sorted(expected - set(supervised))}")
    if supervised["target"].isna().any():
        raise ValueError("Training labels cannot be missing")
    if not supervised["WsDateTime"].is_monotonic_increasing:
        raise ValueError("Supervised timestamps must be sorted")
    if not (supervised["target_time"] == supervised["WsDateTime"] + INTERVAL).all():
        raise ValueError("Target must be the next 15-minute interval")
    n = len(supervised)
    n_test = int(np.ceil(n * test_share))
    train_end = n - n_test - gap_intervals
    if train_end < 100 or n_test < 100:
        raise ValueError("Not enough observations for a reliable chronological split")
    train = supervised.iloc[:train_end].copy()
    test = supervised.iloc[-n_test:].copy()
    # Labels and future feature windows cannot cross the training/test boundary.
    if train["target_time"].max() >= test["WsDateTime"].min():
        raise ValueError("Train/test target leakage across the embargo")
    return train, test


def verify_predictions(frame: pd.DataFrame) -> None:
    required = {"WsDateTime", "target_time", "prediction"}
    if missing := required - set(frame):
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    if frame["target_time"].duplicated().any():
        raise ValueError("Duplicate forecast target timestamps")
    if not (frame["target_time"] == frame["WsDateTime"] + INTERVAL).all():
        raise ValueError("Forecasts must target the next 15-minute interval")
    values = pd.to_numeric(frame["prediction"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Invalid forecast predictions")
