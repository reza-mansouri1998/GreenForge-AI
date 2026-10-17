"""Chunked CSV ingestion with explicit timezone interpretation."""

from __future__ import annotations

import csv
from contextlib import nullcontext
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


def year_bounds(year, timezone="Europe/Berlin", mode="utc_instant"):
    """Half-open calendar-year range; wall-clock mode matches publisher's naive grid."""
    if mode not in {"utc_instant", "publisher_wall_clock"}:
        raise ValueError(f"Unknown timestamp mode: {mode}")
    tz = "UTC" if mode == "publisher_wall_clock" else timezone
    return tuple(int(pd.Timestamp(f"{y}-01-01", tz=tz).timestamp() * 1000) for y in (year, year + 1))


def publisher_name(name):
    """The publisher standardizes numbered channels and V->U voltage prefixes."""
    name = re.sub(r"^\d{3}_", "", name.removesuffix("_EMPTY"))
    # Only voltage-family prefixes, never arbitrary words (e.g. Voltage metadata).
    return re.sub(r"^(THD_)?V(?=[123N]|12|23|31)", lambda m: (m.group(1) or "") + "U", name)


def parse_wall_clock(strings, *, unit="ms"):
    """Preserve publisher wall-clock labels, normalized to the requested unit."""
    if unit not in {"ms", "ns"}:
        raise ValueError("Timestamp unit must be ms or ns")
    strings = strings.astype('string').str.strip()
    aware = strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", na=False)
    if aware.any():
        raise ValueError("Publisher wall-clock mode requires naive timestamps; explicit offsets need review")
    parsed = pd.to_datetime(strings, format="mixed", errors="coerce")
    invalid = int(parsed.isna().sum())
    # Pandas may infer datetime64[s], [ms], [us] or [ns] depending on its
    # version and input precision. astype(int64) returns the *stored* unit,
    # so normalize first. This previously produced 1e9-scale tick_ms values
    # instead of 1e12-scale values and collapsed millions of measurements.
    # The epoch values encode naive wall-clock labels, NOT UTC instants.
    nanoseconds = parsed.dt.as_unit("ns").astype("int64").to_numpy()
    return (nanoseconds if unit == "ns" else nanoseconds // 1_000_000), {
        "invalid_timestamps": invalid, "ambiguous_or_nonexistent_local_times": 0,
    }



def parse_timestamps(strings, timezone, *, unit="ms"):
    """Explicit offsets win; quarantine ambiguous/nonexistent local times."""
    if unit not in {"ms", "ns"}:
        raise ValueError("Timestamp unit must be ms or ns")
    strings = strings.astype("string").str.strip()
    aware = strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", na=False)
    result = pd.Series(pd.NaT, index=strings.index, dtype="datetime64[ns, UTC]")
    result.loc[aware] = pd.to_datetime(strings.loc[aware], format="mixed", utc=True, errors="coerce")
    naive = pd.to_datetime(strings.loc[~aware], format="mixed", errors="coerce")
    bad = int(naive.isna().sum() + result.loc[aware].isna().sum())
    local = naive.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    dst = int(local.isna().sum() - naive.isna().sum())
    result.loc[~aware] = local
    nanoseconds = result.dt.as_unit("ns").astype("int64").to_numpy()
    return (nanoseconds if unit == "ns" else nanoseconds // 1_000_000), {
        "invalid_timestamps": bad,
        "ambiguous_or_nonexistent_local_times": dst,
    }


def align_to_five_seconds(raw_ns):
    """Vectorized pandas-compatible 5-second rounding; ties go to even.

    Inputs are non-null epoch nanoseconds (UTC instants or publisher wall-clock
    labels encoded on a naive epoch). Return millisecond grid ticks and exact
    signed nanosecond offsets. Never use floating point for grid assignment.
    """
    raw_ns = np.asarray(raw_ns, dtype=np.int64)
    quotient, remainder = np.divmod(raw_ns, 5_000_000_000)
    advance = (remainder > 2_500_000_000) | (
        (remainder == 2_500_000_000) & ((quotient & 1) != 0)
    )
    tick_ms = (quotient + advance.astype(np.int64)) * 5_000
    shift_ns = raw_ns - tick_ms * 1_000_000
    return tick_ms, shift_ns



def _publisher_quicksort_tie_winners(absolute_distance_ns, candidate_rows):
    """Replicate the notebook's sort_values('abs_diff')/drop_duplicates tie order.

    Pandas' default quicksort is *unstable*. An earlier/later tie rule does
    not reproduce it. Sort the complete vector of absolute distances, then
    inspect only ordinals that participate in a tied *minimum* for a grid
    tick. The distance vector is written in source-row order and can be a
    read-only memory map, so raw values never need to be loaded into RAM.

    candidate_rows contains (tick_ms, raw_source_ordinal) pairs. Invalid
    timestamps are encoded as int64.max and filtered, matching the notebook's
    drop of NaT timestamps before sort_values().
    """
    if not candidate_rows:
        return {}
    distances = np.asarray(absolute_distance_ns)
    chosen_by_ordinal = {}
    for tick, ordinal in candidate_rows:
        ordinal = int(ordinal)
        if ordinal < 0 or ordinal >= len(distances):
            raise ValueError("Tie candidate ordinal outside source row range")
        chosen_by_ordinal[ordinal] = int(tick)
    possible = np.zeros(len(distances), dtype=np.bool_)
    possible[np.fromiter(chosen_by_ordinal, dtype=np.int64)] = True
    valid = distances != np.iinfo(np.int64).max
    # Crucial: pandas sorts a TimedeltaArray, NOT its integer nanoseconds.
    # NumPy's quicksort tie permutations differ for int64 vs timedelta64.
    # Reinterpret without a copy to match the notebook's sorting dtype.
    timedeltas = distances.view("timedelta64[ns]")
    if np.all(valid):
        sorted_ordinals = np.argsort(timedeltas, kind="quicksort")
    else:
        retained = np.flatnonzero(valid)
        sorted_ordinals = retained[np.argsort(timedeltas[retained], kind="quicksort")]
    # Only a few thousand ties typically survive; avoid a 6-million-entry
    # winner dictionary and keep the original out-of-core DuckDB selection.
    winners = {}
    for ordinal in sorted_ordinals[possible[sorted_ordinals]]:
        tick = chosen_by_ordinal[int(ordinal)]
        winners.setdefault(tick, int(ordinal))
    if len(winners) != len(set(chosen_by_ordinal.values())):
        raise RuntimeError("Publisher tie ordering omitted a candidate tick")
    return winners


def _publisher_tie_candidate_rows(con, source):
    """Find only equidistant *closest* candidates in the existing Parquet."""
    return con.execute(f"""
        WITH best AS (
            SELECT tick_ms, MIN(ABS(shift_ns)) AS best_distance
            FROM {source}
            GROUP BY tick_ms
            HAVING COUNT(*) > 1
        ), nearest AS (
            SELECT r.tick_ms, r.ordinal
            FROM {source} r
            JOIN best b ON r.tick_ms=b.tick_ms
                       AND ABS(r.shift_ns)=b.best_distance
        )
        SELECT tick_ms, ordinal FROM nearest
        QUALIFY COUNT(*) OVER (PARTITION BY tick_ms)>1
    """).fetchall()


def discover(directory, year):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing input directory: {directory}")
    paths = {}
    candidates = set(directory.rglob(f"{year}_*.csv*"))
    if year == 2024 and "PV" in directory.name.upper():
        # Some unformatted PV exports use EnMaSy_###_measurement files.
        candidates.update(directory.rglob("EnMaSy_*.csv*"))
    for path in sorted(candidates):
        if not path.name.endswith((".csv", ".csv.xz")):
            continue
        original = re.sub(r"\.csv(?:\.xz)?$", "", path.name)
        match = re.fullmatch(rf"{year}_(.+)", original)
        legacy_pv = re.fullmatch(r"EnMaSy_\d{3}_(.+)", original)
        if not match and not legacy_pv:
            continue
        name = publisher_name((match or legacy_pv).group(1))
        if name.endswith(("_stats", "_missing")):
            continue
        if name in paths:
            raise ValueError(f"Multiple input files for {name}")
        paths[name] = path
    if not paths:
        raise FileNotFoundError(f"No {year} measurements in {directory}")
    return paths


def normalize_sensor(path, destination, timezone, cfg):
    stat = path.stat()
    mode = cfg.get("timestamp_mode", "utc_instant")
    is_placeholder = "_EMPTY" in path.stem or "_EMPTY" in path.parent.name
    source_hash = digest(path)
    signature = hashlib.sha256(
        json.dumps(
            {"source": source_hash, "timezone": timezone, "timestamp_mode": mode,
             "placeholder": is_placeholder, "parser": digest(Path(__file__))}, sort_keys=True
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
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        sep = ","
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
        "publisher_empty_placeholder": is_placeholder,
        "timestamp_mode": mode,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    unsorted = destination.with_suffix(".unsorted.parquet")
    distance_path = destination.with_suffix(".publisher_distance.bin")
    tie_path = destination.with_suffix(".publisher_ties.parquet")
    # Calculate rounding in NumPy before writing the intermediate Parquet.
    # The previous SQL expression silently collapsed an entire source batch
    # into one output timestamp on a user's DuckDB installation.
    schema = pa.schema([
        ("tick_ms", pa.int64()), ("shift_ns", pa.int64()),
        ("value", pa.float64()), ("ordinal", pa.int64()),
    ])
    max_batch_unique_ticks = 0
    # Allow boundary-adjacent readings but reject wrong time units promptly.
    plausible_start, _ = year_bounds(cfg["year"] - 1, timezone, mode)
    _, plausible_end = year_bounds(cfg["year"] + 1, timezone, mode)
    plausible_start_ns, plausible_end_ns = plausible_start * 1_000_000, plausible_end * 1_000_000
    checked_timestamp_units = False
    try:
        with (stage(f"ingest:{path.name}"),
              pq.ParquetWriter(unsorted, schema, compression="zstd") as writer,
              (open(distance_path, "wb") if mode == "publisher_wall_clock" and not is_placeholder
               else nullcontext(None)) as distance_writer):
            if stat.st_size:
                for frame in pd.read_csv(
                    path,
                    sep=sep,
                    dtype="string",
                    usecols=[time_col, values[0]],
                    encoding="utf-8-sig",
                    chunksize=cfg["batch_rows"],
                ):
                    ticks, problems = (
                        parse_wall_clock(frame[time_col], unit="ns") if mode == "publisher_wall_clock"
                        else parse_timestamps(frame[time_col], timezone, unit="ns")
                    )
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
                    # `is_placeholder` is a Python bool. Do not use `~` on it:
                    # ~False == -1, yielding an *integer index array* and
                    # selecting ticks[1] thousands of times instead of filtering.
                    # Keep this mask boolean for both timestamp and value arrays.
                    valid = ticks != np.iinfo(np.int64).min
                    if is_placeholder:
                        valid[:] = False
                    if valid.dtype != np.dtype('bool'):
                        raise TypeError("Timestamp selection must be a boolean mask")
                    if not checked_timestamp_units and np.any(valid):
                        # Fail before expensive sorting when datetime units
                        # or the source year are not plausible.
                        observed = ticks[valid]
                        if not np.any((observed >= plausible_start_ns) & (observed < plausible_end_ns)):
                            raise ValueError(
                                f"Timestamp-unit/year mismatch in {path}: "
                                "parsed nanosecond timestamps are outside the expected "
                                f"{cfg['year']} measurement period. Check datetime resolution."
                            )
                        checked_timestamp_units = True
                    rounded_ms, shifts_ns = align_to_five_seconds(ticks[valid])
                    if len(rounded_ms):
                        # The full output must contain at least every unique
                        # timestamp seen in a single input chunk.
                        max_batch_unique_ticks = max(
                            max_batch_unique_ticks, int(np.unique(rounded_ms).size)
                        )
                    if distance_writer is not None:
                        # All source ordinals, including invalid timestamps,
                        # must keep their original array positions. This
                        # compact 8-byte-per-row file is deleted after use.
                        abs_distance = np.full(len(frame), np.iinfo(np.int64).max,
                                               dtype=np.int64)
                        abs_distance[valid] = np.abs(shifts_ns)
                        abs_distance.tofile(distance_writer)
                    writer.write_table(
                        pa.Table.from_arrays(
                            [
                                pa.array(rounded_ms),
                                pa.array(shifts_ns),
                                pa.array(numbers[valid], mask=~finite[valid]),
                                pa.array(ordinal[valid]),
                            ],
                            schema=schema,
                        )
                    )
        with connection(destination.parent / "spill", cfg["memory_limit"], cfg["threads"]) as con:
            source = f"read_parquet({literal(unsorted)})"
            # Same instant iff grid tick AND precise fractional offset match.
            report["conflicting_exact_timestamps"] = con.execute(f"""SELECT count(*) FROM (
                SELECT tick_ms, shift_ns FROM {source}
                GROUP BY tick_ms, shift_ns HAVING count(DISTINCT value)>1)""").fetchone()[0]
            # The publisher notebook sorts *all* absolute timestamp offsets
            # with Pandas' default (unstable) quicksort and then drops
            # duplicates. That can select EITHER sign on an exact-distance
            # tie, depending on the surrounding source rows. Do not hardcode
            # earlier or later. Reproduce only this tiny source-row priority
            # detail; DuckDB still performs the scalable nearest-row reduce.
            ranked_source = source
            tie_order = "abs(shift_ns), ordinal"  # original UTC behavior
            report["publisher_equal_distance_tie_ticks"] = 0
            if mode == "publisher_wall_clock" and not is_placeholder and report["source_rows"]:
                candidate_rows = _publisher_tie_candidate_rows(con, source)
                if candidate_rows:
                    distances = np.memmap(distance_path, dtype=np.int64, mode="r")
                    if len(distances) != report["source_rows"]:
                        raise RuntimeError("Publisher distance index is incomplete")
                    winners = _publisher_quicksort_tie_winners(distances, candidate_rows)
                    del distances
                    report["publisher_equal_distance_tie_ticks"] = len(winners)
                    pq.write_table(pa.table({
                        "tick_ms": pa.array(list(winners), type=pa.int64()),
                        "preferred_ordinal": pa.array(list(winners.values()), type=pa.int64()),
                    }), tie_path, compression="zstd")
                    ranked_source = (
                        f"(SELECT r.tick_ms,r.value,r.shift_ns,r.ordinal, "
                        f"CASE WHEN r.ordinal = p.preferred_ordinal THEN 0 ELSE 1 END AS priority "
                        f"FROM {source} r LEFT JOIN read_parquet({literal(tie_path)}) p "
                        f"ON r.tick_ms=p.tick_ms)"
                    )
                    tie_order = "abs(shift_ns), priority, ordinal"
            query = f"""SELECT tick_ms, value,
                       CAST(shift_ns / 1000000.0 AS SMALLINT) AS shift_ms
                       FROM (
                           SELECT tick_ms, value, shift_ns,
                                  row_number() OVER (
                                      PARTITION BY tick_ms
                                      ORDER BY {tie_order}
                                  ) AS row_rank
                           FROM {ranked_source}
                       ) WHERE row_rank=1 ORDER BY tick_ms"""
            with atomic_path(destination) as temp:
                con.execute(
                    f"COPY ({query}) TO {literal(temp)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)"
                )
        report["aligned_rows"] = pq.read_metadata(destination).num_rows
        if report["aligned_rows"] < max_batch_unique_ticks:
            # Mathematical invariant: union of batches includes every tick
            # found in any particular batch. Detect batch-sized collapse.
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Alignment/deduplication lost ticks for {path}: "
                f"output={report['aligned_rows']:,} < "
                f"maximum distinct ticks in one batch={max_batch_unique_ticks:,}. "
                "Refusing to publish a corrupted sensor."
            )
        if report["aligned_rows"] and not is_placeholder:
            # Protect against a SQL rounding/unit regression even if parsing
            # itself was valid. Expected measurement year must overlap.
            start_ms, end_ms = year_bounds(cfg["year"], timezone, mode)
            with connection(destination.parent / "spill", cfg["memory_limit"], cfg["threads"]) as con:
                year_rows = con.execute(
                    f"SELECT count(*) FROM read_parquet({literal(destination)}) "
                    "WHERE tick_ms >= ? AND tick_ms < ?", [start_ms, end_ms]
                ).fetchone()[0]
            if year_rows == 0:
                destination.unlink(missing_ok=True)
                raise ValueError(
                    f"No {cfg['year']} timestamps after alignment for {path}. "
                    "Refusing to publish out-of-range tick_ms values."
                )
        report["collisions_removed"] = max(0,
            report["source_rows"] - report["invalid_timestamps"]
            - report["ambiguous_or_nonexistent_local_times"]
            - (report["source_rows"] if is_placeholder else report["aligned_rows"])
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
        distance_path.unlink(missing_ok=True)
        tie_path.unlink(missing_ok=True)..
