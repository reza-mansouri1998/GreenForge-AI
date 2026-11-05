"""Publish EDA and ML tables after cleaning and target-quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from greenforge_ai.features.interaction_features import interval_values
from greenforge_ai.features.time_series_features import INTERVAL_MS, make_training_frame
from greenforge_ai.utils.streaming import atomic_path, validate_parquet


def _coarser_eda(frame, minutes, target_verified):
    """Combine complete 15-minute observations without filling measurement gaps.

    Preserve the existing EDA column conventions. An interval's mean power
    and energy are only published when every five-second target reading exists.
    """
    if minutes not in (30, 60):
        raise ValueError("Coarser EDA interval must be 30 or 60 minutes")
    blocks = minutes // 15
    cadence = f"{minutes}min"
    source = frame.set_index("WsDateTime")
    output = source["observed_samples"].resample(cadence).sum().to_frame()
    counts = output["observed_samples"]
    observed = "observed_mean_w" if target_verified else "observed_mean_unverified"
    power = "power_w" if target_verified else "power_unverified"
    weighted = (source[observed].fillna(0.0) * source["observed_samples"])
    output[observed] = weighted.resample(cadence).sum() / counts.where(counts > 0)
    # Every 15-minute input is already null when any of its 180 samples is absent.
    full_mean = source[power].resample(cadence).sum(min_count=blocks) / blocks
    output[power] = full_mean.where(counts == 180 * blocks)
    output["coverage"] = counts / (180 * blocks)
    for column in ("Air_Temperature_C", "DayAhead_Price_EUR_MWh"):
        if column in source:
            hourly = source[column].resample(cadence)
            # Do not invent external values for partially observed periods.
            output[column] = hourly.mean().where(hourly.count() == blocks)
    if target_verified:
        output["power_kw"] = output[power] / 1000
        output["energy_kwh"] = output["power_kw"] * (minutes / 60)
        if "DayAhead_Price_EUR_MWh" in output:
            output["energy_value_eur"] = (
                output["energy_kwh"] * output["DayAhead_Price_EUR_MWh"] / 1000
            )
    return output.reset_index()


def make_tables(aggregate, external, system, cfg, directory, target_verified, quality_passed):
    frame = aggregate.finish()
    ticks = frame.pop("tick_ms").to_numpy(dtype=np.int64)
    publisher_mode = cfg.get("timestamp_mode") == "publisher_wall_clock"
    frame.insert(0, "WsDateTime", pd.to_datetime(ticks, unit="ms", utc=not publisher_mode))
    frame["coverage"] = frame.observed_samples / 180
    if "DWD" in external:
        frame["Air_Temperature_C"] = interval_values(external["DWD"], ticks)
    if "SMARD" in external:
        frame["DayAhead_Price_EUR_MWh"] = interval_values(external["SMARD"], ticks)
    if target_verified:
        frame["power_kw"] = frame.power_w / 1000
        frame["energy_kwh"] = frame.power_kw * 0.25
        if "DayAhead_Price_EUR_MWh" in frame:
            frame["energy_value_eur"] = frame.energy_kwh / 1000 * frame.DayAhead_Price_EUR_MWh
    else:
        frame = frame.rename(
            columns={"power_w": "power_unverified", "observed_mean_w": "observed_mean_unverified"}
        )
    with atomic_path(directory / "eda_15min.parquet") as temp:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temp, compression="zstd")
        validate_parquet(temp, step_ms=INTERVAL_MS)
    # Keep the established 15-minute EDA and ML files unchanged; publish
    # additional 30-minute and 1-hour EDA tables using the same measurements.
    for minutes in (30, 60):
        aggregated = _coarser_eda(frame, minutes, target_verified)
        with atomic_path(directory / f"eda_{minutes}min.parquet") as temp:
            pq.write_table(pa.Table.from_pandas(aggregated, preserve_index=False), temp,
                           compression="zstd")
            validate_parquet(temp, step_ms=minutes * 60_000)
    if not target_verified or not quality_passed:
        return {"eda_rows": len(frame), "ml_rows": 0, "ml_status": "blocked_by_quality_review"}
    ml = make_training_frame(frame, external, cfg)
    with atomic_path(directory / "ml.parquet") as temp:
        pq.write_table(pa.Table.from_pandas(ml, preserve_index=False), temp, compression="zstd")
        validate_parquet(temp, dense=True)
    return {
        "eda_rows": len(frame),
        "ml_rows": len(ml),
        "ml_status": "ready" if len(ml) else "no_complete_training_windows",
        "split_counts": ml.split.value_counts().to_dict(),
        "window_diagnostics": ml.attrs.get("window_diagnostics", {}),
    }
