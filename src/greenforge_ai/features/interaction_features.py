"""Exact external interval matching and read-only EDA/plant helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from greenforge_ai.ingestion.dwd_client import HOUR


def interval_values(table, grid, delay_ms=0):
    """Only the matching hour; delay historical weather in ML by one full hour."""
    ticks = table.column(0).cast(pa.int64()).to_numpy() + delay_ms
    values = table.column(1).to_numpy(zero_copy_only=False)
    positions = np.searchsorted(ticks, grid, side="right") - 1
    selected = np.clip(positions, 0, len(ticks) - 1)
    valid = (positions >= 0) & (grid - ticks[selected] < HOUR)
    return np.where(valid, values[selected], np.nan)


def project_root(start=None):
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "etl_config.json").exists():
            return candidate
    raise FileNotFoundError("Run inside the project containing etl_config.json")


def manifest(root=None):
    root = project_root(root)
    config = json.loads((root / "etl_config.json").read_text())
    data = Path(config["data_root"])
    processed = (data if data.is_absolute() else root / data) / "processed"
    return processed, json.loads((processed / "manifest.json").read_text())


def artifact(system, filename, root=None):
    processed, run = manifest(root)
    key = f"{system}/{filename}"
    if key not in run["artifacts"]:
        raise FileNotFoundError(
            f"No {key}; quality status: {run['reports'].get(system, {}).get('features', {})}"
        )
    entry = run["artifacts"][key]
    path = (processed / entry["path"]).resolve()
    if not path.is_relative_to(processed.resolve()):
        raise ValueError("Invalid artifact path")
    stat = path.stat()
    if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
        raise RuntimeError("ETL artifact changed after publication; never overwrite it from a notebook")
    return path


def load_eda(system, root=None):
    return pq.read_table(artifact(system, "eda_15min.parquet", root)).to_pandas()


def scan_measurements(system, columns, start, end, root=None):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not columns or start.tz is None or end.tz is None or end <= start:
        raise ValueError("Choose columns and ordered timezone-aware dates")
    dataset = ds.dataset(artifact(system, "cleaned.parquet", root), format="parquet")
    condition = (ds.field("WsDateTime") >= start.to_pydatetime()) & (
        ds.field("WsDateTime") < end.to_pydatetime()
    )
    return dataset.scanner(
        columns=list(dict.fromkeys(["WsDateTime", *columns])),
        filter=condition,
        batch_size=16384,
        use_threads=False,
    ).to_batches()


def plant_comparison(root=None):
    """Exact matching, not an asserted physical connection between two facilities."""
    machine, solar = load_eda("TEC_48S", root), load_eda("IPE_PV", root)
    if "power_kw" not in machine or "power_kw" not in solar:
        raise ValueError("Plant analysis requires resolved machine AND solar power units")
    out = machine[["WsDateTime", "power_kw", "DayAhead_Price_EUR_MWh"]].merge(
        solar[["WsDateTime", "power_kw"]],
        on="WsDateTime",
        how="inner",
        validate="one_to_one",
        suffixes=("_machine", "_solar"),
    )
    out["net_power_kw"] = out.power_kw_machine - out.power_kw_solar
    out["grid_import_kwh"] = out.net_power_kw.clip(lower=0) * 0.25
    out["grid_export_kwh"] = (-out.net_power_kw).clip(lower=0) * 0.25
    out["import_cost_eur"] = out.grid_import_kwh / 1000 * out.DayAhead_Price_EUR_MWh
    out["complete_interval"] = np.isfinite(
        out[["power_kw_machine", "power_kw_solar", "DayAhead_Price_EUR_MWh"]]
    ).all(axis=1)
    return out
