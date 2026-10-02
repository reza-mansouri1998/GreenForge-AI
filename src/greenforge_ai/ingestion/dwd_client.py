"""DWD observations and shared normalization of hourly external readings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

HOUR = 3_600_000


def hourly_table(rows, start, end, column, sentinel=None):
    grid = np.arange(start // HOUR * HOUR, ((end + HOUR - 1) // HOUR) * HOUR, HOUR, dtype=np.int64)
    values = np.full(len(grid), np.nan)
    conflicts = np.zeros(len(grid), dtype=bool)
    duplicates = 0
    for timestamp, raw in rows:
        if timestamp is None or int(timestamp) != timestamp or int(timestamp) % HOUR:
            raise ValueError("Expected exact hourly UTC external timestamps")
        pos = (int(timestamp) - grid[0]) // HOUR
        if not 0 <= pos < len(grid) or raw is None:
            continue
        value = float(raw)
        if not np.isfinite(value) or value == sentinel:
            continue
        if np.isfinite(values[pos]):
            duplicates += 1
            conflicts[pos] |= not np.isclose(values[pos], value, rtol=0, atol=1e-9)
        else:
            values[pos] = value
    values[conflicts] = np.nan
    if not np.isfinite(values).any():
        raise ValueError(f"No usable {column} observations")
    table = pa.table(
        {
            "WsDateTime": pa.array(grid, type=pa.timestamp("ms", tz="UTC")),
            column: pa.array(values, mask=~np.isfinite(values), type=pa.float32()),
        }
    )
    return table, {
        "rows": len(grid),
        "missing_hours": int((~np.isfinite(values)).sum()),
        "conflicting_hours": int(conflicts.sum()),
        "duplicates": duplicates,
    }


def fetch_dwd(start, end, station):
    from wetterdienst import Settings
    from wetterdienst.provider.dwd.observation import DwdObservationRequest

    request = DwdObservationRequest(
        parameters=[("hourly", "temperature_air", "temperature_air_mean_2m")],
        start_date=pd.Timestamp(start, unit="ms", tz="UTC").isoformat(),
        end_date=pd.Timestamp(end, unit="ms", tz="UTC").isoformat(),
        settings=Settings(
            ts_convert_units=False,
            ts_drop_nulls=False,
            fsspec_client_kwargs={"timeout": 30, "trust_env": True},
        ),
    ).filter_by_station_id(station_id=(str(station).zfill(5),))
    frame = request.values.all().df
    raw = frame.to_arrow() if hasattr(frame, "to_arrow") else pa.Table.from_pandas(frame)
    if raw.num_rows > 30000:
        raise ValueError("Unexpected DWD station/year response size")
    rows = []
    for row in raw.to_pylist():
        if str(row.get("station_id", station)).zfill(5) != str(station).zfill(5):
            continue
        if row["parameter"] not in {"temperature_air_mean_2m", "temperature_air_mean"}:
            continue
        stamp = pd.Timestamp(row["date"])
        if stamp.tz is None:
            raise ValueError("DWD timestamp is missing its timezone")
        rows.append((int(stamp.timestamp() * 1000), row["value"]))
    return hourly_table(rows, start, end, "Air_Temperature_C", sentinel=-999)
