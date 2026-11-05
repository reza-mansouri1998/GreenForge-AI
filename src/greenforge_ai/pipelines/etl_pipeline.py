"""Run: python -m greenforge_ai.pipelines.etl_pipeline --config etl_config.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from greenforge_ai.ingestion.dwd_client import fetch_dwd
from greenforge_ai.ingestion.smard_client import fetch_smard
from greenforge_ai.ingestion.xz_to_parquet import discover, normalize_sensor, year_bounds
from greenforge_ai.transformations.clean_telemetry import build_system
from greenforge_ai.utils.logger import LOG, setup_logging, stage
from greenforge_ai.utils.streaming import (
    atomic_path,
    digest,
    pipeline_lock,
    publish_files,
    recover_publication,
    write_json,
)


def load_external(root, start, end, cfg, offline=False, refresh=False):
    result = {}
    for provider, fetch in [
        ("DWD", lambda: fetch_dwd(start, end, cfg["dwd_station_id"])),
        ("SMARD", lambda: fetch_smard(start, end, cfg["http_workers"])),
    ]:
        key = {
            "provider": provider,
            "start": start,
            "end": end,
            "version": 2,
            "station": cfg["dwd_station_id"] if provider == "DWD" else "DE/4169/hour",
        }
        path, meta = root / f"{provider}.parquet", root / f"{provider}.json"
        cached = json.loads(meta.read_text()) if meta.exists() else {}
        valid = path.exists() and cached.get("key") == key and cached.get("sha256") == digest(path)
        if refresh or not valid:
            if offline:
                raise FileNotFoundError(f"No valid {provider} cache; first run requires API access")
            with stage(f"fetch:{provider}"):
                table, report = fetch()
                with atomic_path(path) as temp:
                    pq.write_table(table, temp, compression="zstd")
                write_json(meta, {"key": key, "sha256": digest(path), "quality": report})
                LOG.info("%s quality=%s", provider, report)
        else:
            LOG.info("CACHE external=%s", provider)
        result[provider] = pq.read_table(path)
    return result


def publisher_local_external(external_utc, year, timezone):
    """Match hourly UTC APIs to the publisher's naive local five-second grid.

    Only the two existing cleaned-output control columns are enriched. The
    Berlin spring-gap and autumn repeated hours have no unique UTC instant,
    so those whole local hours remain null rather than guessing a DST fold.
    """
    columns = {"DWD": "Air_Temperature_C", "SMARD": "DayAhead_Price_EUR_MWh"}
    if set(external_utc) != set(columns):
        raise ValueError("Both DWD and SMARD hourly sources are required")
    local_hours = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left"
    )
    localized = local_hours.tz_localize(
        timezone, ambiguous="NaT", nonexistent="NaT"
    )
    invalid_dst = np.asarray(localized.isna(), dtype=bool)
    local_ticks = local_hours.as_unit("ms").asi8
    utc_ticks = np.full(len(local_hours), np.iinfo(np.int64).min, dtype=np.int64)
    valid_dst = ~invalid_dst
    utc_ticks[valid_dst] = (
        localized[valid_dst].tz_convert("UTC").as_unit("ms").asi8
    )
    result = {}
    for provider, column in columns.items():
        source = external_utc[provider]
        if source.column_names != ["WsDateTime", column]:
            raise ValueError(f"Unexpected {provider} schema: {source.column_names}")
        stamp_type = source.schema.field("WsDateTime").type
        if (not pa.types.is_timestamp(stamp_type)
                or stamp_type.tz != "UTC" or stamp_type.unit != "ms"):
            raise ValueError(f"{provider} time must be UTC milliseconds")
        ticks = source.column("WsDateTime").cast(pa.int64()).to_numpy(zero_copy_only=False)
        if not len(ticks) or np.any(np.diff(ticks) <= 0):
            raise ValueError(f"{provider} hourly timestamps must be sorted and unique")
        if np.any(ticks % 3_600_000):
            raise ValueError(f"{provider} timestamps must be exact UTC hours")
        values = source.column(column).to_numpy(zero_copy_only=False)
        data = np.full(len(local_hours), np.nan, dtype=np.float32)
        query = utc_ticks[valid_dst]
        positions = np.searchsorted(ticks, query)
        found = positions < len(ticks)
        indices = np.minimum(positions, len(ticks) - 1)
        found &= ticks[indices] == query
        data[np.flatnonzero(valid_dst)[found]] = np.asarray(
            values[indices[found]], dtype=np.float32
        )
        result[provider] = pa.table({
            "WsDateTime": pa.array(local_ticks, type=pa.timestamp("ms")),
            column: pa.array(data, mask=~np.isfinite(data), type=pa.float32()),
        })
        LOG.info(
            "%s Berlin-local API hours: valid=%d missing=%d DST_unresolvable=%d",
            provider, int(np.isfinite(data).sum()), int((~np.isfinite(data)).sum()),
            int(invalid_dst.sum()),
        )
    return result


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "data_root",
        "year",
        "source_timezone",
        "timestamp_mode",
        "external_enrichment",
        "memory_limit",
        "threads",
        "batch_rows",
        "dwd_station_id",
        "http_workers",
        "history_intervals",
        "horizon_intervals",
        "systems",
    }
    if set(cfg) - allowed:
        raise ValueError(f"Unknown/obsolete configuration keys: {set(cfg) - allowed}")
    cfg.setdefault("timestamp_mode", "utc_instant")
    if cfg["timestamp_mode"] not in {"utc_instant", "publisher_wall_clock"}:
        raise ValueError("timestamp_mode must be utc_instant or publisher_wall_clock")
    cfg.setdefault("external_enrichment", "off")
    if cfg["external_enrichment"] not in {"off", "safe_local_hour"}:
        raise ValueError("external_enrichment must be off or safe_local_hour")
    # The enrichment mode applies to publisher wall-clock data. UTC-instant
    # runs retain the existing direct UTC join for regression compatibility.
    if not 2018 <= cfg["year"] <= 2024:
        raise ValueError("This historical SPARK/SMARD hourly profile supports 2018–2024")
    if not 1024 <= cfg["batch_rows"] <= 65536:
        raise ValueError("batch_rows must be 1024–65536")
    if not 1 <= cfg["threads"] <= 4 or not 1 <= cfg["http_workers"] <= 4:
        raise ValueError("Limit SQL/HTTP workers to 1–4")
    if not 2 <= cfg["history_intervals"] <= 96 or not 1 <= cfg["horizon_intervals"] <= 96:
        raise ValueError("Invalid feature history/forecast horizon")
    ids = [s["id"] for s in cfg["systems"]]
    if len(ids) != len(set(ids)) or any(not n.replace("_", "").isalnum() for n in ids):
        raise ValueError("System IDs must be unique alphanumeric/underscore names")
    for system in cfg["systems"]:
        if system["kind"] not in {"machine", "pv"}:
            raise ValueError("Unknown system kind")
        if (
            system["rated_current_a"] <= 0
            or system["line_voltage_v"] <= 0
            or not 1 <= system["capacity_margin"] <= 2
        ):
            raise ValueError("Invalid machine capacity configuration")
    root = Path(cfg["data_root"])
    cfg["data_root"] = str((path.parent / root).resolve()) if not root.is_absolute() else str(root)
    # Make schema paths independent of the calling working directory, and
    # ensure the file whose hash is recorded is the one build_system reads.
    for system in cfg["systems"]:
        if ref := system.get("publisher_schema_file"):
            schema = Path(ref)
            if not schema.is_absolute():
                schema = (path.parent / schema).resolve()
            if not schema.is_file():
                raise FileNotFoundError(f"Publisher schema not found for {system['id']}: {schema}")
            system["publisher_schema_file"] = str(schema)
    year_bounds(cfg["year"], cfg["source_timezone"], cfg["timestamp_mode"])
    return cfg


def publisher_schema_hashes(cfg):
    """Include schema content in incremental-cache provenance, not just its path."""
    return {
        system["id"]: digest(Path(system["publisher_schema_file"]))
        for system in cfg["systems"]
        if system.get("publisher_schema_file")
    }


def run(config_path, offline=False, refresh_external=False, force=False):
    cfg = load_config(config_path.resolve())
    root = Path(cfg["data_root"])
    processed = root / "processed"
    setup_logging(config_path.resolve().parent / "logs")
    with pipeline_lock(processed / "etl.lock"), stage("ETL"):
        recover_publication(processed)
        start, end = year_bounds(cfg["year"], cfg["source_timezone"], cfg["timestamp_mode"])
        # Fetch the original UTC API hours into the existing data/external cache.
        # Only external values are projected to naive local hours; the machine
        # and PV measurement grid (and all cleaning rules) remain unchanged.
        if cfg["timestamp_mode"] == "publisher_wall_clock":
            if cfg["external_enrichment"] == "safe_local_hour":
                utc_start, utc_end = year_bounds(
                    cfg["year"], cfg["source_timezone"], "utc_instant"
                )
                api_data = load_external(
                    root / "external", utc_start - 3_600_000, utc_end, cfg,
                    offline=offline, refresh=refresh_external,
                )
                external = publisher_local_external(
                    api_data, cfg["year"], cfg["source_timezone"]
                )
            else:
                external = {}
        else:
            external = load_external(
                root / "external", start - 3_600_000, end, cfg,
                offline=offline, refresh=refresh_external,
            )
        sources, signatures, input_reports = {}, {}, {}
        for system in cfg["systems"]:
            paths = {}
            for name, path in discover(root / system["input_dir"], cfg["year"]).items():
                destination = root / "raw" / "extracted" / system["id"] / f"{cfg['year']}_{name}.parquet"
                report = normalize_sensor(path, destination, cfg["source_timezone"], cfg)
                key = f"{system['id']}/{name}"
                signatures[key], input_reports[key], paths[name] = (
                    report["output_sha256"],
                    report,
                    destination,
                )
            sources[system["id"]] = paths
        for provider in external:
            signatures[provider] = digest(root / "external" / f"{provider}.parquet")
        code_root = Path(__file__).resolve().parents[1]
        code_hashes = {p.relative_to(code_root).as_posix(): digest(p) for p in code_root.rglob("*.py")}
        schema_hashes = publisher_schema_hashes(cfg)
        payload = {"config": cfg, "sources": signatures, "code": code_hashes,
                   "publisher_schema_hashes": schema_hashes}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        manifest_path = processed / "manifest.json"
        if manifest_path.exists() and not force:
            old = json.loads(manifest_path.read_text())
            if old.get("fingerprint") == fingerprint and all(
                (processed / e["path"]).exists() and digest(processed / e["path"]) == e["sha256"]
                for e in old.get("artifacts", {}).values()
            ):
                LOG.info("UNCHANGED: reusing verified outputs")
                return old
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = Path(tempfile.mkdtemp(prefix=".staging_", dir=processed))
        reports = {}
        try:
            for system in cfg["systems"]:
                reports[system["id"]] = build_system(
                    system, sources[system["id"]], external, start, end, cfg, run_dir / system["id"]
                )
                if cfg["timestamp_mode"] == "publisher_wall_clock" and external:
                    # build_system already filled the two existing control columns.
                    # Correct its legacy status label in both reports and quality.json.
                    reports[system["id"]]["external_join_status"] = "enabled_safe_local_hour"
                    write_json(
                        run_dir / system["id"] / "quality.json", reports[system["id"]]
                    )
            artifacts, staged = {}, {}
            for path in run_dir.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    relative = path.relative_to(run_dir)
                    category = "features" if path.name in {
                        "eda_15min.parquet", "eda_30min.parquet", "eda_60min.parquet", "ml.parquet"
                    } else "cleaned"
                    target = processed / category / relative
                    staged[target] = path
                    artifacts[relative.as_posix()] = {
                        "path": target.relative_to(processed).as_posix(),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": digest(path),
                    }
            manifest = {
                "run_id": run_id,
                "fingerprint": fingerprint,
                "config": cfg,
                "artifacts": artifacts,
                "reports": reports,
                "input_reports": input_reports,
                "source_hashes": signatures,
                "code_hashes": code_hashes,
                "publisher_schema_hashes": schema_hashes,
            }
            # Remove a previous ML file when the new target fails quality gates.
            for system in cfg["systems"]:
                target = processed / "features" / system["id"] / "ml.parquet"
                if target not in staged and target.exists():
                    staged[target] = None
            staged_manifest = run_dir / "manifest.json"
            write_json(staged_manifest, manifest)
            staged[manifest_path] = staged_manifest
            publish_files(processed, staged)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        for name, report in reports.items():
            if report["features"]["ml_status"] != "ready":
                LOG.warning("%s ML NOT READY: %s; see quality.json", name, report["features"]["ml_status"])
        LOG.info("Published run=%s. Missing measurements remain missing by design.", run_id)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("etl_config.json"))
    parser.add_argument("--offline", action="store_true", help="Require valid cached API data")
    parser.add_argument("--refresh-external", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.config, offline=args.offline, refresh_external=args.refresh_external, force=args.force)


if __name__ == "__main__":
    main()
