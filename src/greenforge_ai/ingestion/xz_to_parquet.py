"""Chunked CSV ingestion with explicit timezone interpretation."""

from __future__ import annotations

import csv
import hashlib
import json
import lzma
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from greenforge_ai.utils.logger import LOG, stage
from greenforge_ai.utils.streaming import atomic_path, connection, digest, literal, write_json

SCHEMA = pa.schema([("tick_ms", pa.int64()), ("value", pa.float64()), ("shift_ms", pa.int16())])


def year_bounds(year, timezone="Europe/Berlin"):
    return tuple(int(pd.Timestamp(f"{y}-01-01", tz=timezone).timestamp() * 1000) for y in (year, year + 1))


def parse_timestamps(strings, timezone):
    """Explicit offsets win; ambiguous/nonexistent naive local times are quarantined."""
    strings = strings.astype("string").str.strip()
    aware = strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", na=False)
    result = pd.Series(pd.NaT, index=strings.index, dtype="datetime64[ns, UTC]")
    result.loc[aware] = pd.to_datetime(strings.loc[aware], format="mixed", utc=True, errors="coerce")
    naive = pd.to_datetime(strings.loc[~aware], format="mixed", errors="coerce")
    bad = int(naive.isna().sum() + result.loc[aware].isna().sum())
    local = naive.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    dst = int(local.isna().sum() - naive.isna().sum())
    result.loc[~aware] = local
    return result.astype("int64").to_numpy() // 1_000_000, {
        "invalid_timestamps": bad,
        "ambiguous_or_nonexistent_local_times": dst,
    }


def discover(directory, year):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing input directory: {directory}")
    paths = {}
    for path in sorted(directory.rglob(f"{year}_*.csv*")):
        if not path.name.endswith((".csv", ".csv.xz")):
            continue
        name = re.sub(r"\.csv(?:\.xz)?$", "", path.name)[5:]
        if name.endswith(("_stats", "_missing")):
            continue
        name = name.removesuffix("_EMPTY")
        if name in paths:
            raise ValueError(f"Multiple input files for {name}")
        paths[name] = path
    if not paths:
        raise FileNotFoundError(f"No {year} measurements in {directory}")
    return paths


def normalize_sensor(path, destination, timezone, cfg):
    stat = path.stat()
    source_hash = digest(path)
    signature = hashlib.sha256(
        json.dumps(
            {"source": source_hash, "timezone": timezone, "parser": digest(Path(__file__))}, sort_keys=True
        ).encode()
    ).hexdigest()
    meta = destination.with_suffix(".json")
    if destination.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old.get("signature") == signature and old.get("output_sha256") == digest(destination):
            LOG.info("CACHE sensor=%s", path.name)
            return old
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
    if not sample.strip():
        sample = "WsDateTime,value\n"
    sep = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    headers = next(csv.reader(sample.splitlines(), delimiter=sep))
    time_columns = [c for c in headers if c.strip().lower() in {"wsdatetime", "timestamp", "time", "date"}]
    if len(headers) != len(set(headers)) or len(time_columns) != 1:
        raise ValueError(f"Ambiguous header/time schema: {path}: {headers}")
    time_col = time_columns[0]
    values = [c for c in headers if c != time_col and not c.lower().startswith("unnamed:")]
    if len(values) != 1:
        raise ValueError(f"Expected one measurement column: {path}")
    report = {
        "source": str(path),
        "source_sha256": source_hash,
        "signature": signature,
        "source_rows": 0,
        "invalid_timestamps": 0,
        "ambiguous_or_nonexistent_local_times": 0,
        "invalid_numeric_values": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    unsorted = destination.with_suffix(".unsorted.parquet")
    schema = pa.schema([("raw_ms", pa.int64()), ("value", pa.float64()), ("ordinal", pa.int64())])
    try:
        with stage(f"ingest:{path.name}"), pq.ParquetWriter(unsorted, schema, compression="zstd") as writer:
            if stat.st_size:
                for frame in pd.read_csv(
                    path,
                    sep=sep,
                    dtype="string",
                    usecols=[time_col, values[0]],
                    encoding="utf-8-sig",
                    chunksize=cfg["batch_rows"],
                ):
                    ticks, problems = parse_timestamps(frame[time_col], timezone)
                    for k, v in problems.items():
                        report[k] += v
                    numbers = pd.to_numeric(frame[values[0]], errors="coerce").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                    finite = np.isfinite(numbers)
                    report["invalid_numeric_values"] += int(
                        (~finite & frame[values[0]].notna().to_numpy()).sum()
                    )
                    ordinal = np.arange(report["source_rows"], report["source_rows"] + len(frame))
                    report["source_rows"] += len(frame)
                    valid = ticks != np.iinfo(np.int64).min // 1_000_000
                    writer.write_table(
                        pa.Table.from_arrays(
                            [
                                pa.array(ticks[valid]),
                                pa.array(numbers[valid], mask=~finite[valid]),
                                pa.array(ordinal[valid]),
                            ],
                            schema=schema,
                        )
                    )
        with connection(destination.parent / "spill", cfg["memory_limit"], cfg["threads"]) as con:
            source = f"read_parquet({literal(unsorted)})"
            report["conflicting_exact_timestamps"] = con.execute(f"""SELECT count(*) FROM (
                SELECT raw_ms FROM {source} GROUP BY raw_ms HAVING count(DISTINCT value)>1)""").fetchone()[0]
            query = f"""WITH timed AS (
                SELECT *, CAST(floor(raw_ms/5000.0+0.5) AS BIGINT)*5000 AS tick_ms FROM {source}
            ), ranked AS (
                SELECT *, row_number() OVER (PARTITION BY tick_ms ORDER BY abs(raw_ms-tick_ms),ordinal) AS rank
                FROM timed
            ) SELECT tick_ms,value,CAST(raw_ms-tick_ms AS SMALLINT) AS shift_ms
              FROM ranked WHERE rank=1 ORDER BY tick_ms"""
            with atomic_path(destination) as temp:
                con.execute(
                    f"COPY ({query}) TO {literal(temp)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)"
                )
        report["aligned_rows"] = pq.read_metadata(destination).num_rows
        report["collisions_removed"] = (
            report["source_rows"]
            - report["invalid_timestamps"]
            - report["ambiguous_or_nonexistent_local_times"]
            - report["aligned_rows"]
        )
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            destination.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            raise RuntimeError(f"Source changed during ingestion: {path}")
        report["output_sha256"] = digest(destination)
        write_json(meta, report)
        if report["invalid_timestamps"] or report["ambiguous_or_nonexistent_local_times"]:
            LOG.warning("Timestamp quarantine: %s", report)
        return report
    finally:
        unsorted.unlink(missing_ok=True)
