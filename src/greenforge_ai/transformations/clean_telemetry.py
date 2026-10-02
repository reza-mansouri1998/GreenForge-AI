"""Preserve measured gaps and partially observed sensors with traceable quality reports."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from greenforge_ai.features.interaction_features import interval_values
from greenforge_ai.features.time_series_features import QuarterHour
from greenforge_ai.pipelines.feature_pipeline import make_tables
from greenforge_ai.transformations.consolidate_telemetry import SensorCursor
from greenforge_ai.transformations.validate import (
    PV_REVIEW_CHANNELS,
    REFERENCE_CHANNELS,
    Relationships,
    inconsistent_power,
    rule_for,
)
from greenforge_ai.utils.logger import LOG, stage
from greenforge_ai.utils.streaming import atomic_path, validate_parquet, write_json


class GapTracker:
    def __init__(self, sensor, writer):
        self.sensor, self.writer, self.start = sensor, writer, None
        self.count, self.longest = 0, 0

    def update(self, ticks, missing):
        if self.start is not None and not missing[0]:
            self.close(int(ticks[0]))
        edges = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(np.int8)))
        for lo, hi in zip(edges[::2], edges[1::2]):
            if self.start is None:
                self.start = int(ticks[lo])
            if hi < len(ticks):
                self.close(int(ticks[hi]))

    def close(self, end):
        if self.start is not None:
            duration = (end - self.start) // 1000
            self.writer.writerow([self.sensor, self.start, end, duration])
            self.count += 1
            self.longest = max(self.longest, duration)
            self.start = None


def assess_current_scale(paths, start, end, cfg):
    names = ["I_sys", "I1", "I2", "I3"]
    if not all(n in paths for n in names):
        return None, {"status": "not_tested", "reason": "missing_phase_channels"}
    support, hits = 0, {0.001: 0, 1.0: 0, 1000.0: 0}
    with ExitStack() as stack:
        cursors = {n: SensorCursor(paths[n]) for n in names}
        for cursor in cursors.values():
            stack.callback(cursor.close)
        for low in range(start, end, cfg["batch_rows"] * 5000):
            grid = np.arange(low, min(end, low + cfg["batch_rows"] * 5000), 5000, dtype=np.int64)
            data = {n: cursors[n].read_grid(grid)[0] for n in names}
            reference, current = (data["I1"] + data["I2"] + data["I3"]) / 3, data["I_sys"]
            mask = np.isfinite(reference) & np.isfinite(current) & (reference > 0.1) & (current > 0)
            a, b = current[mask], reference[mask]
            support += len(a)
            for factor in hits:
                hits[factor] += int((abs(a * factor - b) <= 0.02 * b).sum())
    candidates = [f for f, n in hits.items() if support >= 256 and n / support >= 0.99]
    factor = candidates[0] if len(candidates) == 1 else None
    return factor, {
        "support_rows": support,
        "matching_rows_by_factor": hits,
        "selected_factor": factor,
        "status": "cross_signal_supported" if factor is not None else "unresolved",
    }


def build_system(system, paths, external, start, end, cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    exclusions = {}
    if system["kind"] == "machine":
        exclusions = {n: "Relative fundamental-reference channel; originals retained"
                      for n in set(paths) & REFERENCE_CHANNELS}
    elif system["id"] in {"IPE_PV", "EPI_PV"} and system["input_profile"] == "spark_raw":
        exclusions = {n: "Unverified raw PV register mapping; originals retained for review"
                      for n in set(paths) & PV_REVIEW_CHANNELS
                      if n not in system.get("scale_factors", {})}
    excluded = sorted(exclusions)
    if system["target"] in excluded:
        raise ValueError("An excluded review channel cannot be the prediction target")
    paths = {name: path for name, path in paths.items() if name not in excluded}
    if excluded:
        LOG.info("%s excluded analytical channels (originals retained): %s", system["id"], excluded)
    names, target = sorted(paths), system["target"]
    if target not in names:
        raise ValueError(f"Missing required target {target}")
    extra = (set(system.get("scale_factors", {})) | set(system.get("limits", {}))) - set(names)
    if extra:
        raise ValueError(f"Configured rules name missing channels: {extra}")
    system = {**system, "scale_factors": dict(system.get("scale_factors", {}))}
    scale_report = {"status": "not_needed"}
    if (
        system["input_profile"] == "spark_raw"
        and system["kind"] == "machine"
        and "I_sys" not in system["scale_factors"]
    ):
        factor, scale_report = assess_current_scale(paths, start, end, cfg)
        if factor is not None:
            system["scale_factors"]["I_sys"] = factor
    rules = {n: rule_for(n, system) for n in names}
    unknown = [n for n, r in rules.items() if r.status.startswith("unverified")]
    if unknown:
        LOG.warning("%s unverified channels retained for review: %s", system["id"], unknown)
    controls = [
        "Air_Temperature_C",
        "DayAhead_Price_EUR_MWh",
        "row_has_missing",
        "row_has_outlier",
        "row_was_time_aligned",
    ]
    if set(names) & {"WsDateTime", *controls}:
        raise ValueError("Measurement uses a reserved output name")
    schema = pa.schema(
        [
            ("WsDateTime", pa.timestamp("ms", tz="UTC")),
            *[(n, pa.float32()) for n in names],
            *[(n, pa.float32()) for n in controls[:2]],
            *[(n, pa.bool_()) for n in controls[2:]],
        ],
        metadata={
            b"timezone": b"UTC",
            b"gap_policy": b"no_imputation",
            b"source_timezone_for_naive_values": cfg["source_timezone"].encode(),
            b"measurement_rules": json.dumps({n: r.metadata() for n, r in rules.items()}).encode(),
        },
    )
    totals = {
        n: {"valid": 0, "missing": 0, "outliers": 0, "sum": 0.0, "min": None, "max": None} for n in names
    }
    relationships, aggregate = Relationships(), QuarterHour()
    last_daily, last_day = None, None
    rejected_power = 0
    path = directory / "cleaned.parquet"
    with ExitStack() as stack, stage(f"clean:{system['id']}"):
        cursors = {n: SensorCursor(paths[n]) for n in names}
        for cursor in cursors.values():
            stack.callback(cursor.close)
        gap_handle = stack.enter_context((directory / "gaps.csv").open("w", newline="", encoding="utf-8"))
        gap_writer = csv.writer(gap_handle)
        gap_writer.writerow(
            ["sensor", "start_utc_epoch_ms", "end_exclusive_utc_epoch_ms", "duration_seconds"]
        )
        gaps = {n: GapTracker(n, gap_writer) for n in names}
        with (
            atomic_path(path) as temp,
            pq.ParquetWriter(temp, schema, compression="zstd", use_dictionary=False) as writer,
        ):
            for index, low in enumerate(range(start, end, cfg["batch_rows"] * 5000)):
                grid = np.arange(low, min(end, low + cfg["batch_rows"] * 5000), 5000, dtype=np.int64)
                arrays, data = [pa.array(grid, type=pa.timestamp("ms", tz="UTC"))], {}
                missing = np.zeros(len(grid), dtype=bool)
                outlier, shifted = missing.copy(), missing.copy()
                for name in names:
                    raw, shift = cursors[name].read_grid(grid)
                    values, bad = rules[name].clean(raw)
                    if name == "P_total" and all(n in data for n in ("P1", "P2", "P3")):
                        rejected = inconsistent_power(values, [data[n] for n in ("P1", "P2", "P3")])
                        rejected_power += int(rejected.sum())
                        bad |= rejected
                        values[rejected] = np.nan
                    if name == "DailyYield" and name not in unknown:
                        days = pd.to_datetime(grid, unit="ms", utc=True).tz_convert("Europe/Berlin").date
                        for i in np.flatnonzero(np.isfinite(values)):
                            value, day = values[i], days[i]
                            if last_day == day and last_daily is not None and value < last_daily:
                                bad[i], values[i] = True, np.nan
                            if not bad[i]:
                                last_daily, last_day = value, day
                    valid = np.isfinite(values)
                    stat = totals[name]
                    stat["valid"] += int(valid.sum())
                    stat["missing"] += int((~valid).sum())
                    stat["outliers"] += int(bad.sum())
                    if valid.any():
                        v = values[valid]
                        stat["sum"] += float(v.sum())
                        stat["min"] = (
                            float(v.min()) if stat["min"] is None else min(stat["min"], float(v.min()))
                        )
                        stat["max"] = (
                            float(v.max()) if stat["max"] is None else max(stat["max"], float(v.max()))
                        )
                    gaps[name].update(grid, ~valid)
                    data[name] = values
                    missing |= ~valid
                    outlier |= bad
                    shifted |= shift != 0
                    arrays.append(pa.array(values, mask=~valid, type=pa.float32()))
                for provider in ("DWD", "SMARD"):
                    values = interval_values(external[provider], grid)
                    missing |= ~np.isfinite(values)
                    arrays.append(pa.array(values, mask=~np.isfinite(values), type=pa.float32()))
                arrays.extend(pa.array(a) for a in (missing, outlier, shifted))
                writer.write_table(
                    pa.Table.from_arrays(arrays, schema=schema), row_group_size=cfg["batch_rows"]
                )
                relationships.update({n: a for n, a in data.items() if n not in unknown})
                aggregate.update(grid, data[target])
                if index % 32 == 0:
                    LOG.info(
                        "%s clean_progress=%.1f%%",
                        system["id"],
                        100 * (grid[-1] + 5000 - start) / (end - start),
                    )
            for tracker in gaps.values():
                tracker.close(end)
        validation = validate_parquet(path, step_ms=5000)
    for name, stat in totals.items():
        total = stat.pop("sum")
        stat["mean"] = total / stat["valid"] if stat["valid"] else None
        stat["structurally_zero"] = stat["valid"] > 0 and stat["min"] == stat["max"] == 0
        stat["gap_count"], stat["longest_gap_seconds"] = gaps[name].count, gaps[name].longest
        LOG.info("%s sensor=%s valid=%d missing=%d rejected=%d longest_gap_s=%d",
                 system["id"], name, stat["valid"], stat["missing"], stat["outliers"],
                 stat["longest_gap_seconds"])
    if rejected_power:
        LOG.warning("%s rejected %d inconsistent total-power readings; source values retained",
                    system["id"], rejected_power)
    relations = relationships.report()
    stat = totals[target]
    fraction = stat["outliers"] / max(1, stat["valid"] + stat["outliers"])
    critical = relations["checks"].get("P_phase_sum", {}) if target == "P_total" else {}
    passed = stat["valid"] > 0 and fraction <= 0.05 and critical.get("status") != "review_required"
    features = make_tables(aggregate, external, system, cfg, directory, target not in unknown, passed)
    report = {
        "system": system["id"],
        "rows": validation["rows"],
        "gap_policy": "no_imputation",
        "excluded_channels": exclusions,
        "target_consistency": {
            "check": "P_total versus coincident P1 + P2 + P3",
            "rejected_rows": rejected_power,
            "policy": "Reject inconsistent total readings; never reconstruct or fill them",
            "relative_tolerance": 0.2,
            "absolute_tolerance_w": 1.0,
        } if target == "P_total" else {"status": "not_applicable"},
        "rules": {n: r.metadata() for n, r in rules.items()},
        "unverified_channels": unknown,
        "I_sys_scale_assessment": scale_report,
        "sensors": totals,
        "relationships": relations,
        "target_outlier_fraction": fraction,
        "target_quality_passed": bool(passed and target not in unknown),
        "features": features,
        "status": "ready_for_review"
        if unknown or not passed or relations["status"] != "passed"
        else "checks_passed",
        "source_policy": "Publisher-informed; not certified identical to the official cleaned archive",
    }
    write_json(directory / "quality.json", report)
    return report
