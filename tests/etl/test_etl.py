"""Time, physics, gap, merge, forecasting and publication regression tests."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from greenforge_ai.features.interaction_features import artifact, interval_values
from greenforge_ai.features.time_series_features import QuarterHour, make_training_frame
from greenforge_ai.ingestion.dwd_client import hourly_table
from greenforge_ai.ingestion.xz_to_parquet import SCHEMA, normalize_sensor, parse_timestamps, year_bounds
from greenforge_ai.pipelines.etl_pipeline import load_config
from greenforge_ai.transformations.clean_telemetry import GapTracker, assess_current_scale, build_system
from greenforge_ai.transformations.consolidate_telemetry import SensorCursor
from greenforge_ai.transformations.validate import rule_for
from greenforge_ai.utils.streaming import atomic_path, pipeline_lock, validate_parquet


def test_atomic_publication_uses_writable_descriptor(tmp_path, monkeypatch):
    """Simulate Windows' writable-descriptor requirement on any platform."""
    real_fsync = os.fsync
    synced = []

    def writable_fsync(fd):
        os.write(fd, b"")  # Raises EBADF for a read-only descriptor on POSIX.
        real_fsync(fd)
        synced.append(fd)

    monkeypatch.setattr(os, "fsync", writable_fsync)
    path = tmp_path / "output.bin"
    path.write_bytes(b"previous")
    with atomic_path(path) as temp:
        temp.write_bytes(b"complete replacement")
    assert path.read_bytes() == b"complete replacement"
    assert len(synced) == 1
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_sync_failure_preserves_previous(tmp_path, monkeypatch):
    def failed_fsync(fd):
        raise OSError(errno.EIO, "Simulated disk sync failure")

    monkeypatch.setattr(os, "fsync", failed_fsync)
    path = tmp_path / "output.bin"
    path.write_bytes(b"previous")
    with pytest.raises(OSError, match="Simulated disk sync failure"):
        with atomic_path(path) as temp:
            temp.write_bytes(b"replacement")
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.fixture
def cfg():
    value = json.loads((Path(__file__).parents[2] / "etl_config.json").read_text())
    value["timestamp_mode"] = "utc_instant"  # Regression tests for original behavior
    # These tests deliberately construct reduced sensor sets. Keep production
    # publisher projection and candidate limits in the dedicated schema tests;
    # synthetic builds must only validate rules for channels they provide.
    tec = next(system for system in value["systems"] if system["id"] == "TEC_48S")
    tec["limits"] = {}
    tec.pop("publisher_schema_file", None)
    # Production may intentionally process TEC_48S alone. Keep synthetic PV
    # coverage in the test suite without changing production etl_config.json.
    if not any(system["id"] == "IPE_PV" for system in value["systems"]):
        value["systems"].append({
            "id": "IPE_PV", "kind": "pv", "input_dir": "raw/spark_raw/IPE_PV",
            "input_profile": "spark_raw", "target": "AC_ActivePower",
            "rated_current_a": 16.0, "line_voltage_v": 400.0,
            "capacity_margin": 1.1, "scale_factors": {}, "limits": {},
        })
    return value


def sensor(path, ticks, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(ticks, type=pa.int64()),
                pa.array(values, type=pa.float64(), from_pandas=True),
                pa.array(np.zeros(len(ticks), dtype=np.int16)),
            ],
            schema=SCHEMA,
        ),
        path,
    )
    return path


def external(start, end):
    times = range(start // 3_600_000 * 3_600_000 - 3_600_000, end, 3_600_000)
    return {
        "DWD": hourly_table([(t, 12.0) for t in times], start - 3_600_000, end, "Air_Temperature_C")[0],
        "SMARD": hourly_table([(t, -50.0) for t in times], start - 3_600_000, end, "DayAhead_Price_EUR_MWh")[
            0
        ],
    }


def test_offsets_and_local_times():
    times, report = parse_timestamps(
        pd.Series(["2024-01-01 01:00:00", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00+01:00"]),
        "Europe/Berlin",
    )
    assert len(set(times)) == 1 and sum(report.values()) == 0


def test_dst_ambiguity_is_quarantined():
    _, report = parse_timestamps(pd.Series(["2024-03-31 02:30:00", "2024-10-27 02:30:00"]), "Europe/Berlin")
    assert report["ambiguous_or_nonexistent_local_times"] == 2
    times, _ = parse_timestamps(
        pd.Series(["2024-10-27T02:30:00+02:00", "2024-10-27T02:30:00+01:00"]), "Europe/Berlin"
    )
    assert times[1] - times[0] == 3_600_000


def test_calendar_year_is_utc_safe():
    start, end = year_bounds(2024)
    assert pd.Timestamp(start, unit="ms", tz="UTC") == pd.Timestamp("2023-12-31T23:00:00Z")
    assert (end - start) // 5000 == 366 * 24 * 720


def test_nearest_tick_closest_and_cache(tmp_path, cfg):
    path = tmp_path / "2024_P_total.csv"
    path.write_text(
        "WsDateTime,P_total\n2024-01-01T00:00:02Z,10\n2024-01-01T00:00:00Z,20\n"
        "2024-01-01T00:00:04Z,30\n2024-01-01T00:00:05Z,40\n2024-01-01T00:00:05Z,50\n"
    )
    output = tmp_path / "sensor.parquet"
    report = normalize_sensor(path, output, "Europe/Berlin", cfg)
    assert pq.read_table(output)["value"].to_pylist() == [20.0, 40.0]
    assert report["collisions_removed"] == 3 and report["conflicting_exact_timestamps"] == 1
    mtime = output.stat().st_mtime_ns
    normalize_sensor(path, output, "Europe/Berlin", cfg)
    assert output.stat().st_mtime_ns == mtime


def test_sensor_cursor_preserves_gaps(tmp_path):
    c = SensorCursor(sensor(tmp_path / "s.parquet", [0, 10000, 20000], [1.0, 2.0, 3.0]))
    try:
        first, _ = c.read_grid(np.arange(0, 15000, 5000))
        second, _ = c.read_grid(np.arange(15000, 25000, 5000))
        np.testing.assert_allclose(first, [1, np.nan, 2], equal_nan=True)
        np.testing.assert_allclose(second, [np.nan, 3], equal_nan=True)
    finally:
        c.close()


def test_signed_power_limits(cfg):
    values, bad = rule_for("P1", cfg["systems"][0]).clean(np.array([-10.0, 100.0, 1e9]))
    assert values[0] == -10 and np.isnan(values[-1]) and bad.sum() == 1


def test_frequency_limits(cfg):
    values, _ = rule_for("Freq", cfg["systems"][0]).clean(np.array([48.0, 50.0, 52.0]))
    np.testing.assert_allclose(values, [np.nan, 50, np.nan], equal_nan=True)


def test_raw_and_cleaned_scale_profiles(cfg):
    machine = cfg["systems"][0]
    assert rule_for("THD_I1", machine).factor == 0.1
    assert rule_for("THD_I1", {**machine, "input_profile": "spark_cleaned"}).factor == 1
    assert rule_for("I_sys", machine).status == "unverified_scale"


def test_documented_pv_scale_and_explicit_override(cfg):
    pv = cfg["systems"][1]
    rule = rule_for("AC_ActivePower", pv)
    assert rule.status == "publisher_raw_pv_kw_to_w" and rule.factor == 1000
    assert rule_for("AC_ActivePower", {**pv, "input_profile": "spark_cleaned"}).factor == 1
    assert rule_for("DC_Power_1", pv).status.startswith("unverified")
    configured = {**pv, "scale_factors": {"AC_ActivePower": 1000}}
    assert rule_for("AC_ActivePower", configured).clean(np.array([2.0]))[0][0] == 2000


def test_gap_spans_batches():
    class Collector:
        def __init__(self):
            self.rows = []

        def writerow(self, row):
            self.rows.append(row)

    writer = Collector()
    tracker = GapTracker("P", writer)
    tracker.update(np.array([0, 5000, 10000]), np.array([False, True, True]))
    tracker.update(np.array([15000, 20000]), np.array([True, False]))
    tracker.close(25000)
    assert writer.rows == [["P", 5000, 20000, 15]]


def test_external_no_imputation_conflicts_negative_prices():
    table, report = hourly_table(
        [(0, -20.0), (0, -20.0), (7200000, 10.0), (7200000, 11.0)], 0, 10800000, "price"
    )
    assert report["conflicting_hours"] == 1
    values = interval_values(table, np.array([0, 3599999, 3600000, 7200000, 99000000]))
    np.testing.assert_allclose(values, [-20, -20, np.nan, np.nan, np.nan], equal_nan=True)


def test_weather_sentinel_and_availability():
    table, _ = hourly_table([(0, 10.0), (3600000, -999)], 0, 7200000, "temperature", -999)
    values = interval_values(table, np.array([0, 3600000, 7200000]), delay_ms=3600000)
    np.testing.assert_allclose(values, [np.nan, 10, np.nan], equal_nan=True)


def test_quarter_energy_and_partial_rejection():
    q = QuarterHour()
    ticks, values = np.arange(0, 1800000, 5000), np.full(360, 4000.0)
    values[-1] = np.nan
    q.update(ticks[:190], values[:190])
    q.update(ticks[190:], values[190:])
    frame = q.finish()
    assert frame.observed_samples.tolist() == [180, 179]
    assert frame.power_w.iloc[0] / 1000 * 0.25 == 1 and np.isnan(frame.power_w.iloc[1])


def test_forecast_target_and_gap_boundaries(cfg):
    frame = pd.DataFrame(
        {
            "WsDateTime": pd.date_range("1970-01-01", periods=100, freq="15min", tz="UTC"),
            "power_kw": np.arange(100, dtype=float),
        }
    )
    frame.loc[30, "power_kw"] = np.nan
    out = make_training_frame(frame, external(0, 100 * 900000), cfg)
    assert not out.isna().any().any()
    assert (out.target_power_kw == out.power_last_15m_kw + 1).all()
    assert not out.power_last_15m_kw.between(30, 33).any()
    assert (
        out.loc[out.split == "train", "Target_End"].max()
        <= out.loc[out.split == "test", "Feature_Window_Start"].min()
    )


def test_atomic_failure_preserves_previous(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("previous")
    with pytest.raises(RuntimeError), atomic_path(path) as temp:
        temp.write_text("partial")
        raise RuntimeError("simulated failure")
    assert path.read_text() == "previous"


def test_lock_prevents_overlap(tmp_path):
    with pipeline_lock(tmp_path / "lock"), pytest.raises(RuntimeError), pipeline_lock(tmp_path / "lock"):
        pass


def test_current_scale_from_coincident_channels(tmp_path, cfg):
    ticks = np.arange(0, 400 * 5000, 5000)
    paths = {
        n: sensor(tmp_path / f"{n}.parquet", ticks, np.full(400, v))
        for n, v in [("I1", 2), ("I2", 4), ("I3", 6), ("I_sys", 4000)]
    }
    factor, report = assess_current_scale(paths, 0, 400 * 5000, cfg)
    assert factor == 0.001 and report["support_rows"] == 400


def test_partial_auxiliary_sensors_are_preserved(tmp_path, cfg):
    end = 24 * 3600000
    ticks = np.arange(0, end, 5000)
    power = np.full(len(ticks), 3000.0)
    power[10:20] = np.nan
    paths = {
        "P_total": sensor(tmp_path / "P.parquet", ticks, power),
        "U1": sensor(tmp_path / "U.parquet", ticks, np.where(np.arange(len(ticks)) % 2, 230.0, np.nan)),
    }
    report = build_system(cfg["systems"][0], paths, external(0, end), 0, end, cfg, tmp_path / "out")
    assert "U1" in pq.read_schema(tmp_path / "out/cleaned.parquet").names
    assert report["sensors"]["P_total"]["missing"] == 10 and report["features"]["ml_status"] == "ready"
    validate_parquet(tmp_path / "out/ml.parquet", dense=True)


def test_unresolved_solar_blocks_energy_and_ml(tmp_path, cfg):
    ticks = np.arange(0, 3600000, 5000)
    paths = {"AC_ActivePower": sensor(tmp_path / "P.parquet", ticks, np.ones(len(ticks)))}
    report = build_system({**cfg["systems"][1], "id": "UNKNOWN_PV"}, paths, external(0, 3600000), 0, 3600000, cfg, tmp_path / "out")
    assert report["features"]["ml_status"] == "blocked_by_quality_review"
    assert not (tmp_path / "out/ml.parquet").exists()
    assert "energy_kwh" not in pq.read_schema(tmp_path / "out/eda_15min.parquet").names


def test_duplicate_boundary_and_nonfinite_validation(tmp_path):
    path = tmp_path / "bad.parquet"
    times = np.r_[np.arange(16384) * 5000, 16383 * 5000]
    pq.write_table(
        pa.table(
            {"WsDateTime": pa.array(times, type=pa.timestamp("ms", tz="UTC")), "v": np.ones(len(times))}
        ),
        path,
    )
    with pytest.raises(ValueError):
        validate_parquet(path)
    for value in (float("nan"), float("inf")):
        pq.write_table(
            pa.table({"WsDateTime": pa.array([0], type=pa.timestamp("ms", tz="UTC")), "v": [value]}), path
        )
        with pytest.raises(ValueError, match="NaN/infinity"):
            validate_parquet(path)


def test_reject_old_config_flags(tmp_path, cfg):
    cfg["build_unified"] = True
    path = tmp_path / "etl_config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="obsolete"):
        load_config(path)


def test_pipeline_publication_cache_recovery_and_tamper(tmp_path, cfg, monkeypatch):
    import greenforge_ai.pipelines.etl_pipeline as etl

    start = year_bounds(2024)[0]
    end = start + 24 * 3600000
    for system in cfg["systems"]:
        system["input_profile"] = "spark_cleaned"
        folder = tmp_path / "data" / system["input_dir"] / system["target"]
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "WsDateTime": pd.to_datetime(np.arange(start, end, 5000), unit="ms", utc=True),
                system["target"]: 1000.0,
            }
        ).to_csv(folder / f"2024_{system['target']}.csv.xz", index=False)
    config = tmp_path / "etl_config.json"
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(etl, "year_bounds", lambda year, timezone, mode="utc_instant": (start, end))

    def mock_external(root, low, high, config, **kwargs):
        root.mkdir(parents=True, exist_ok=True)
        tables = external(start, end)
        for key, table in tables.items():
            pq.write_table(table, root / f"{key}.parquet")
        return tables

    monkeypatch.setattr(etl, "load_external", mock_external)
    first = etl.run(config)
    for system in cfg["systems"]:
        extracted = tmp_path / "data/raw/extracted" / system["id"]
        assert (extracted / f"2024_{system['target']}.parquet").is_file()
    assert not (tmp_path / "data/cache").exists()
    processed = tmp_path / "data/processed"
    assert not (processed / "runs").exists()
    for system in cfg["systems"]:
        assert (processed / "cleaned" / system["id"] / "cleaned.parquet").is_file()
        assert (processed / "features" / system["id"] / "eda_15min.parquet").is_file()
    assert not list(processed.glob(".staging_*"))
    assert first["reports"]["IPE_PV"]["features"]["ml_status"] == "ready"
    assert etl.run(config)["run_id"] == first["run_id"]
    path = artifact("TEC_48S", "cleaned.parquet", tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(etl, "build_system", fail)
    with pytest.raises(RuntimeError, match="injected"):
        etl.run(config, force=True)
    assert artifact("TEC_48S", "cleaned.parquet", tmp_path) == path
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="changed"):
        artifact("TEC_48S", "cleaned.parquet", tmp_path)


def test_external_cache_identity_and_exact_plant_join(tmp_path, cfg, monkeypatch):
    import greenforge_ai.features.interaction_features as nb
    import greenforge_ai.pipelines.etl_pipeline as ext

    count = []

    def weather(start, end, station):
        count.append(station)
        return hourly_table([(0, 12.0)], start, end, "Air_Temperature_C")

    monkeypatch.setattr(ext, "fetch_dwd", weather)
    monkeypatch.setattr(
        ext,
        "fetch_smard",
        lambda start, end, workers: hourly_table([(0, -1.0)], start, end, "DayAhead_Price_EUR_MWh"),
    )
    ext.load_external(tmp_path, 0, 3600000, cfg)
    ext.load_external(tmp_path, 0, 3600000, cfg, offline=True)
    assert len(count) == 1
    with pytest.raises(FileNotFoundError):
        ext.load_external(tmp_path, 0, 3600000, {**cfg, "dwd_station_id": "00001"}, offline=True)
    machine = pd.DataFrame(
        {
            "WsDateTime": pd.to_datetime(["2024-01-01", "2024-12-01"], utc=True),
            "power_kw": [4.0, 8.0],
            "DayAhead_Price_EUR_MWh": [100.0, 100.0],
        }
    )
    solar = pd.DataFrame({"WsDateTime": pd.to_datetime(["2024-01-01"], utc=True), "power_kw": [2.0]})
    monkeypatch.setattr(nb, "load_eda", lambda name, root: machine if name == "TEC_48S" else solar)
    result = nb.plant_comparison()
    assert len(result) == 1 and result.import_cost_eur.iloc[0] == 0.05


def test_solar_short_coverage_has_evaluation(cfg):
    frame = pd.DataFrame(
        {
            "WsDateTime": pd.date_range("2024-01-01", periods=400, freq="15min", tz="UTC"),
            "power_kw": np.r_[np.ones(120), np.full(280, np.nan)],
        }
    )
    start = int(frame.WsDateTime.iloc[0].timestamp() * 1000)
    result = make_training_frame(frame, external(start, start + 400 * 900000), cfg)
    assert set(result.split) == {"train", "validation", "test"}


def test_empty_sensor_is_all_missing(tmp_path):
    cursor = SensorCursor(sensor(tmp_path / "empty.parquet", [], []))
    try:
        values, shifts = cursor.read_grid(np.arange(0, 20000, 5000))
        assert np.isnan(values).all() and (shifts == 0).all()
    finally:
        cursor.close()


def test_logging_routes_channels_and_repeated_setup(tmp_path):
    from greenforge_ai.utils.logger import LOG, get_logger, setup_logging

    setup_logging(tmp_path)
    setup_logging(tmp_path)
    LOG.info("etl-only-event")
    get_logger("greenforge_ai.training.fit").warning("training-only-event")
    get_logger("app.main").info("application-only-event")
    expected = {
        "pipeline": "etl-only-event",
        "model_training": "training-only-event",
        "application": "application-only-event",
    }
    for filename, marker in expected.items():
        files = list((tmp_path / filename).glob("*.log"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert content.count(marker) == 1
        assert all(other not in content for other in expected.values() if other != marker)
        assert "Z |" in content and "pid=" in content
    with pytest.raises(ValueError, match="Unknown logging channel"):
        get_logger("test", channel="invalid")


def test_logging_rotation_is_bounded(tmp_path):
    from greenforge_ai.utils.logger import get_logger, setup_logging

    setup_logging(tmp_path, max_bytes=200, backup_count=2)
    logger = get_logger("app.test")
    for index in range(20):
        logger.info("event=%s %s", index, "x" * 80)
    files = list((tmp_path / "application").glob("*.log*"))
    assert len(files) == 3
    assert "event=19" in next((tmp_path / "application").glob("*.log")).read_text()


def test_logging_new_runs_preserve_previous_files(tmp_path):
    from greenforge_ai.utils.logger import LOG, setup_logging

    setup_logging(tmp_path, new_run=True)
    LOG.info("first-run-event")
    first = next((tmp_path / "pipeline").glob("*.log"))
    before = first.read_bytes()
    setup_logging(tmp_path, new_run=True)
    LOG.info("second-run-event")
    files = list((tmp_path / "pipeline").glob("*.log"))
    assert len(files) == 2
    assert first.read_bytes() == before
    second = next(path for path in files if path != first)
    assert "second-run-event" in second.read_text()
    assert "first-run-event" not in second.read_text()
    assert "Z_" in second.name
    assert not list((tmp_path / "application").glob("*.log"))



def test_publication_failure_rolls_back_all_files(tmp_path, monkeypatch):
    from greenforge_ai.utils.streaming import publish_files

    processed = tmp_path / "processed"
    processed.mkdir()
    first, second = processed / "first", processed / "second"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"new-first")
    b.write_bytes(b"new-second")
    original = os.replace

    def fail_once(source, target):
        if source == b:
            raise PermissionError("simulated Windows file in use")
        original(source, target)

    monkeypatch.setattr(os, "replace", fail_once)
    with pytest.raises(PermissionError):
        publish_files(processed, {first: a, second: b})
    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert not list(processed.glob(".*"))


def test_publication_recovers_interrupted_run(tmp_path):
    from greenforge_ai.utils.streaming import recover_publication, write_json

    backup = tmp_path / ".backup_test"
    backup.mkdir()
    (backup / "existing").write_bytes(b"old")
    (tmp_path / "existing").write_bytes(b"partially published")
    (tmp_path / "new").write_bytes(b"uncommitted")
    write_json(tmp_path / ".publication.json", {
        "backup": backup.name,
        "entries": [{"path": "existing", "existed": True}, {"path": "new", "existed": False}],
    })
    recover_publication(tmp_path)
    assert (tmp_path / "existing").read_bytes() == b"old"
    assert not (tmp_path / "new").exists()
    assert not backup.exists()
    recover_publication(tmp_path)  # Recovery is safe to repeat.


def test_publication_removes_stale_ml_file(tmp_path):
    from greenforge_ai.utils.streaming import publish_files

    old = tmp_path / "ml.parquet"
    old.write_bytes(b"previous eligible training data")
    publish_files(tmp_path, {old: None})
    assert not old.exists()



def test_reference_channels_excluded_but_rms_retained(tmp_path, cfg):
    ticks = np.arange(0, 3600000, 5000)
    paths = {
        "P_total": sensor(tmp_path / "power.parquet", ticks, np.ones(len(ticks))),
        "U1_fund": sensor(tmp_path / "reference.parquet", ticks, np.full(len(ticks), 1000.0)),
        "U1_RMS_fund": sensor(tmp_path / "rms.parquet", ticks, np.full(len(ticks), 230.0)),
    }
    report = build_system(cfg["systems"][0], paths, external(0, 3600000), 0, 3600000, cfg, tmp_path / "out")
    names = pq.read_schema(tmp_path / "out/cleaned.parquet").names
    assert "U1_fund" not in names and "U1_RMS_fund" in names
    assert "U1_fund" in report["excluded_channels"]
    assert paths["U1_fund"].exists()
    assert report["sensors"]["U1_RMS_fund"]["outliers"] == 0


def test_tec_voltage_harmonic_units_do_not_double_scale(cfg):
    system = cfg["systems"][0]
    for name in ["THD_U1", "THD_U31", "U1_h2", "U31_h5"]:
        raw = rule_for(name, system)
        cleaned = rule_for(name, {**system, "input_profile": "spark_cleaned"})
        assert raw.factor == 0.1 and cleaned.factor == 1
        assert raw.unit == "%" and not raw.status.startswith("unverified")
        assert raw.clean(np.array([200.0]))[0][0] == 20.0
    assert rule_for("U1_RMS_fund", system).unit == "V"
    assert rule_for("I1_fund", system).status == "unverified_reference_channel"


def test_pv_raw_kw_becomes_watts_in_eda(tmp_path, cfg):
    ticks = np.arange(0, 3600000, 5000)
    paths = {"AC_ActivePower": sensor(tmp_path / "pv.parquet", ticks, np.full(len(ticks), 2.0))}
    report = build_system(cfg["systems"][1], paths, external(0, 3600000), 0, 3600000, cfg, tmp_path / "out")
    eda = pq.read_table(tmp_path / "out/eda_15min.parquet").to_pandas()
    np.testing.assert_allclose(eda.power_kw, 2.0)
    np.testing.assert_allclose(eda.energy_kwh, 0.5)
    assert report["sensors"]["AC_ActivePower"]["max"] == 2000


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_forecast_timestamp_units_match_external(cfg, unit, monkeypatch):
    original_timedelta = pd.Timedelta
    monkeypatch.setattr(pd, "Timedelta", lambda *args, **kwargs: original_timedelta(*args, **kwargs).as_unit(unit))
    frame = pd.DataFrame({
        "WsDateTime": pd.date_range("2024-01-01", periods=120, freq="15min", tz="UTC").as_unit(unit),
        "power_kw": np.ones(120),
    })
    start = int(frame.WsDateTime.iloc[0].timestamp() * 1000)
    result = make_training_frame(frame, external(start, start + 120 * 900000), cfg)
    assert not result.empty
    assert set(result.split) == {"train", "validation", "test"}
    assert (result.previous_hour_temperature_c == 12.0).all()
    assert (result.target_day_ahead_price_eur_mwh == -50.0).all()


def test_inconsistent_total_quarantines_rows_not_whole_year(tmp_path, cfg):
    end = 24 * 3600000
    ticks = np.arange(0, end, 5000)
    total = np.full(len(ticks), 3000.0)
    total[1000:1030] = 67.0
    paths = {"P_total": sensor(tmp_path / "total.parquet", ticks, total)}
    paths.update({f"P{i}": sensor(tmp_path / f"phase{i}.parquet", ticks, np.full(len(ticks), 1000.0))
                  for i in (1, 2, 3)})
    report = build_system(cfg["systems"][0], paths, external(0, end), 0, end, cfg, tmp_path / "out")
    cleaned = pq.read_table(tmp_path / "out/cleaned.parquet")
    assert cleaned["P_total"].null_count == 30
    assert report["target_consistency"]["rejected_rows"] == 30
    assert report["relationships"]["checks"]["P_phase_sum"]["status"] == "passed"
    assert report["features"]["ml_status"] == "ready"
    ml = pq.read_table(tmp_path / "out/ml.parquet").to_pandas()
    assert (ml.target_power_kw == 3.0).all() and not ml.isna().any().any()


def test_power_consistency_preserves_missing_phases_and_zero_noise():
    from greenforge_ai.transformations.validate import inconsistent_power
    mask = inconsistent_power(np.array([0.0, 67.0, 3000.0, 3000.0]),
                              [np.array([0.1, 50.0, 1000.0, np.nan]),
                               np.array([0.1, 17.0, 1000.0, 1000.0]),
                               np.array([0.1, 210.0, 1000.0, 1000.0])])
    assert mask.tolist() == [False, True, False, False]


def test_raw_pv_unsupported_registers_excluded_daily_yield_kept(tmp_path, cfg):
    ticks = np.arange(0, 3600000, 5000)
    paths = {name: sensor(tmp_path / f"{name}.parquet", ticks, np.full(len(ticks), value))
             for name, value in [("AC_ActivePower", 2.0), ("DailyYield", 500.0),
                                 ("GridFreq", 410.0), ("DC_Current_1", 12933897.0),
                                 ("DC_Voltage_1", 42949670.0), ("TotalYield", 800.0)]}
    report = build_system(cfg["systems"][1], paths, external(0, 3600000), 0, 3600000, cfg, tmp_path / "out")
    names = pq.read_schema(tmp_path / "out/cleaned.parquet").names
    assert "DailyYield" in names and "AC_ActivePower" in names
    assert set(report["excluded_channels"]) == {"GridFreq", "DC_Current_1", "DC_Voltage_1", "TotalYield"}
    assert all(path.exists() for path in paths.values())
    assert report["unverified_channels"] == []
    assert rule_for("DailyYield", cfg["systems"][1]).unit == "Wh"


def test_daily_counter_rejects_decrease_without_changing_baseline(tmp_path, cfg):
    ticks = np.arange(0, 20000, 5000)
    paths = {"AC_ActivePower": sensor(tmp_path / "ac.parquet", ticks, np.zeros(4)),
             "DailyYield": sensor(tmp_path / "yield.parquet", ticks, [500, 0, 100, 550])}
    build_system(cfg["systems"][1], paths, external(0, 20000), 0, 20000, cfg, tmp_path / "out")
    assert pq.read_table(tmp_path / "out/cleaned.parquet")["DailyYield"].to_pylist() == [500, None, None, 550]


def test_relationship_violations_do_not_cancel():
    from greenforge_ai.transformations.validate import Relationships
    checks = Relationships()
    checks.add("power", np.array([0.0, 200.0]), np.array([100.0, 100.0]))
    report = checks.report()["checks"]["power"]
    assert report["status"] == "review_required"
    assert report["violation_fraction"] == 1.0


def test_relationship_absolute_floor():
    from greenforge_ai.transformations.validate import Relationships
    checks = Relationships()
    checks.add("power", np.array([0.1]), np.array([0.2]))
    assert checks.report()["status"] == "passed"


def test_uba_conversion_and_missing_energy():
    import polars as pl
    from datetime import date
    from greenforge_ai.features.emissions import add_emissions
    frame = pl.DataFrame({"Date": [date(2024, 1, 1)] * 2, "import_kwh": [10.0, None]})
    result = add_emissions(frame)
    assert result["estimated_import_co2_kg"][0] == pytest.approx(3.63)
    assert result["estimated_import_co2_kg"][1] is None
    assert "estimated_import_co2_kg" not in frame.columns


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_uba_rejects_invalid_energy(value):
    import polars as pl
    from datetime import date
    from greenforge_ai.features.emissions import add_emissions
    with pytest.raises(ValueError):
        add_emissions(pl.DataFrame({"Date": [date(2024, 1, 1)], "import_kwh": [value]}))


def test_uba_year_and_local_timezone():
    import polars as pl
    from datetime import datetime, timezone, date
    from greenforge_ai.features.emissions import add_emissions
    stamp = datetime(2023, 12, 31, 23, tzinfo=timezone.utc)
    assert add_emissions(pl.DataFrame({"Date": [stamp], "import_kwh": [1.0]}))["estimated_import_co2_kg"][0] == pytest.approx(.363)
    with pytest.raises(ValueError):
        add_emissions(pl.DataFrame({"Date": [date(2025, 1, 1)], "import_kwh": [1.0]}))


def test_publisher_mode_uses_full_naive_calendar_grid():
    from greenforge_ai.ingestion.xz_to_parquet import year_bounds, parse_wall_clock
    start, end = year_bounds(2024, mode="publisher_wall_clock")
    assert pd.Timestamp(start, unit="ms") == pd.Timestamp("2024-01-01 00:00:00")
    assert (end - start) // 5000 == 366 * 24 * 720
    values, problems = parse_wall_clock(pd.Series([
        "2024-10-27 02:30:00", "2024-03-31 02:30:00"
    ]))
    assert len(values) == 2 and sum(problems.values()) == 0
    assert pd.Timestamp(values[0], unit="ms") == pd.Timestamp("2024-10-27 02:30:00")
    with pytest.raises(ValueError, match="naive"):
        parse_wall_clock(pd.Series(["2024-01-01T00:00:00Z"]))


def test_publisher_mode_integrates_voltage_f_zero_rejection(tmp_path, cfg):
    from greenforge_ai.transformations.validate import publisher_rule_for

    cfg = {**cfg, "timestamp_mode": "publisher_wall_clock"}
    tec = {**cfg["systems"][0]}
    tec.pop("publisher_availability", None)
    end = 20_000
    ticks = np.arange(0, end, 5_000)
    paths = {
        "P_total": sensor(tmp_path / "P_total.parquet", ticks, [100.0] * len(ticks)),
        "U1_f": sensor(tmp_path / "U1_f.parquet", ticks, [0.0, 230.0, 0.0, 240.0]),
        "I1": sensor(tmp_path / "I1.parquet", ticks, [0.0, 2.0, 0.0, 3.0]),
    }

    report = build_system(tec, paths, external(0, end), 0, end, cfg, tmp_path / "out")
    table = pq.read_table(tmp_path / "out/cleaned.parquet")

    assert table["U1_f"].to_pylist() == [None, 230.0, None, 240.0]
    assert table["I1"].to_pylist() == [0.0, 2.0, 0.0, 3.0]
    assert report["sensors"]["U1_f"]["valid"] == 2
    assert report["sensors"]["U1_f"]["missing"] == 2
    assert report["sensors"]["U1_f"]["outliers"] == 2
    assert report["sensors"]["I1"]["valid"] == 4
    assert report["sensors"]["I1"]["missing"] == 0
    assert not publisher_rule_for("U1_f", {
        "id": "IPE_PV", "input_profile": "spark_raw", "kind": "pv",
        "scale_factors": {}, "limits": {},
    }).reject_zero


def test_publisher_file_name_normalization(tmp_path):
    from greenforge_ai.ingestion.xz_to_parquet import publisher_name, discover
    assert publisher_name("004_V1") == "U1"
    assert publisher_name("009_THD_V12") == "THD_U12"
    assert publisher_name("P_total") == "P_total"
    folder = tmp_path / "EPI_PV" / "008_AC_ActivePower"
    folder.mkdir(parents=True)
    sample = folder / "EnMaSy_008_AC_ActivePower.csv.xz"
    sample.touch()
    assert discover(folder.parent, 2024) == {"AC_ActivePower": sample}


def test_publisher_rules_preserve_power_and_clamp_known_voltage(cfg):
    from greenforge_ai.transformations.validate import publisher_rule_for
    tec = cfg["systems"][0]
    power = publisher_rule_for("P_total", tec)
    assert power.factor == 1.0 and power.unit == "W"
    assert power.lower is None and power.upper is None
    volts = publisher_rule_for("U1", tec)
    cleaned, rejected = volts.clean(np.array([205.0, 230.0, 257.0]))
    np.testing.assert_allclose(cleaned, [np.nan, 230.0, np.nan], equal_nan=True)
    assert rejected.tolist() == [True, False, True]
    assert publisher_rule_for("P_total", {**tec, "input_profile": "spark_cleaned"}).factor == 1


def test_publisher_5s_rounding_ties_to_even_and_empty_placeholder(tmp_path, cfg):
    cfg = {**cfg, "timestamp_mode": "publisher_wall_clock"}
    path = tmp_path / "2024_U1.csv"
    path.write_text("WsDateTime,U1\n2024-01-01 00:00:02.500,220\n"
                    "2024-01-01 00:00:02.500000400,225\n"
                    "2024-01-01 00:00:07.500,230\n", encoding="utf-8")
    dest = tmp_path / "U1.parquet"
    normalized = normalize_sensor(path, dest, "Europe/Berlin", cfg)
    assert normalized["aligned_rows"] == 3
    assert pq.read_table(dest)["tick_ms"].to_pylist() == [
        int(pd.Timestamp("2024-01-01 00:00:00").value // 10**6),
        int(pd.Timestamp("2024-01-01 00:00:05").value // 10**6),
        int(pd.Timestamp("2024-01-01 00:00:10").value // 10**6),
    ]
    placeholder = tmp_path / "2024_U2_EMPTY.csv"
    placeholder.write_text("WsDateTime,U2\n2024-01-01 00:00:00,0.0\n")
    ignored = normalize_sensor(placeholder, tmp_path / "U2.parquet", "Europe/Berlin", cfg)
    assert ignored["publisher_empty_placeholder"]
    assert ignored["aligned_rows"] == 0


def test_publisher_wall_clock_nanosecond_unit_regression(monkeypatch):
    """Pandas may infer datetime64[us]; alignment must still receive raw_ns."""
    import greenforge_ai.ingestion.xz_to_parquet as ingestion

    original = ingestion.pd.to_datetime

    def simulated_microsecond_resolution(*args, **kwargs):
        # Reproduces pandas versions/inputs that infer datetime64[us].
        return original(*args, **kwargs).dt.as_unit("us")

    monkeypatch.setattr(ingestion.pd, "to_datetime", simulated_microsecond_resolution)
    strings = pd.Series(["2024-01-01 00:00:00", "2024-01-01 00:00:05"])
    ns, report = ingestion.parse_wall_clock(strings, unit="ns")
    assert report["invalid_timestamps"] == 0
    assert ns.tolist() == [1704067200000000000, 1704067205000000000]
    ms, _ = ingestion.parse_wall_clock(strings, unit="ms")
    assert ms.tolist() == [1704067200000, 1704067205000]


def test_5_second_alignment_retains_thousands_of_ticks_across_batches(tmp_path, cfg):
    """Regression: never collapse a whole 8192-row batch into one tick."""
    cfg = {**cfg, "timestamp_mode": "publisher_wall_clock", "batch_rows": 1024}
    times = pd.date_range("2024-01-01", periods=2050, freq="5s")
    data = pd.DataFrame({
        "WsDateTime": times.strftime("%Y-%m-%d %H:%M:%S"),
        "P_total": np.arange(len(times), dtype=float),
    }).sample(frac=1, random_state=17)
    raw = tmp_path / "2024_P_total.csv"
    data.to_csv(raw, index=False)
    output = tmp_path / "P_total.parquet"

    report = normalize_sensor(raw, output, "Europe/Berlin", cfg)
    result = pq.read_table(output)
    assert report["source_rows"] == 2050
    assert report["aligned_rows"] == 2050
    assert report["collisions_removed"] == 0
    np.testing.assert_array_equal(
        result["tick_ms"].to_numpy(),
        # Pandas may store DatetimeIndex values as s/us/ns depending on version.
        # Convert to milliseconds explicitly before comparing with tick_ms.
        times.as_unit("ms").astype("int64").to_numpy(),
    )
    np.testing.assert_array_equal(
        result["value"].to_numpy(), np.arange(2050, dtype=float),
    )


def test_5_second_alignment_uses_nearest_reading_with_midpoint_tie_even(tmp_path, cfg):
    from greenforge_ai.ingestion.xz_to_parquet import align_to_five_seconds

    raw = pd.to_datetime(pd.Series([
        "2024-01-01 00:00:00", "2024-01-01 00:00:02.500",
        "2024-01-01 00:00:02.500000400", "2024-01-01 00:00:07.500",
    ]), format="mixed")
    ns = raw.dt.as_unit("ns").astype("int64").to_numpy()
    ticks, shifts = align_to_five_seconds(ns)
    expected = raw.dt.round("5s").dt.as_unit("ms").astype("int64").to_numpy()
    np.testing.assert_array_equal(ticks, expected)
    np.testing.assert_array_equal(shifts, ns - ticks * 1_000_000)

def test_alignment_boolean_mask_never_selects_second_row_repeatedly(tmp_path, cfg, monkeypatch):
    """Regression: `~False` in a NumPy mask became int indices of 1."""
    import greenforge_ai.ingestion.xz_to_parquet as ingestion

    cfg = {**cfg, "timestamp_mode": "publisher_wall_clock", "batch_rows": 1024}
    n = 2049  # More than two ingestion chunks; each row is a unique 5s tick.
    times = pd.date_range("2024-01-10 00:00:00", periods=n, freq="5s")
    source = tmp_path / "2024_P_total.csv"
    pd.DataFrame({
        "WsDateTime": times.strftime("%Y-%m-%d %H:%M:%S"),
        "P_total": np.arange(n, dtype=float),
    }).sample(frac=1, random_state=32).to_csv(source, index=False)
    recorded = []
    original = ingestion.align_to_five_seconds

    def capture(raw_ns):
        # Catch the bug before the SQL step; all input timestamps in this
        # deliberately unique sample must still be different at this point.
        assert len(raw_ns) == np.unique(raw_ns).size
        recorded.append(len(raw_ns))
        return original(raw_ns)

    monkeypatch.setattr(ingestion, "align_to_five_seconds", capture)
    output = tmp_path / "clean.parquet"
    report = ingestion.normalize_sensor(source, output, "Europe/Berlin", cfg)
    assert recorded == [1024, 1024, 1]
    assert report["aligned_rows"] == n
    assert report["collisions_removed"] == 0
    result = pq.read_table(output)
    np.testing.assert_array_equal(result["value"].to_numpy(), np.arange(n, dtype=float))


def test_invalid_datetime_does_not_select_valid_row_by_integer_index(tmp_path, cfg):
    cfg = {**cfg, "timestamp_mode": "publisher_wall_clock"}
    source = tmp_path / "2024_P_total.csv"
    source.write_text(
        "WsDateTime,P_total\n"
        "2024-01-01 00:00:00,1.0\n"
        "not-a-timestamp,99.0\n"
        "2024-01-01 00:00:05,2.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "clean.parquet"
    report = normalize_sensor(source, output, "Europe/Berlin", cfg)
    assert report["invalid_timestamps"] == 1
    assert report["aligned_rows"] == 2
    assert pq.read_table(output)["value"].to_pylist() == [1.0, 2.0]



def test_publisher_pv_rule_supports_raw_folder_identifier(cfg):
    """Renaming ETL output to IPE_PV must not disable raw kW -> W scaling."""
    from greenforge_ai.transformations.validate import publisher_rule_for

    pv = next(s for s in cfg["systems"] if s["id"] == "IPE_PV")
    for system_id in ("IPE_PV", "EPI_PV"):
        system = {**pv, "id": system_id, "input_profile": "spark_raw"}
        power = publisher_rule_for("AC_ActivePower", system)
        assert power.factor == 1000.0 and power.unit == "W"
        assert power.status == "publisher_pv_kw_to_w"
        yield_rule = publisher_rule_for("DailyYield", system)
        assert yield_rule.factor == 1.0 and yield_rule.unit == "Wh"
        assert yield_rule.status == "publisher_daily_yield_wh"
        cleaned, bad = power.clean(np.array([2.5]))
        assert cleaned.tolist() == [2500.0] and bad.tolist() == [False]


def test_publisher_equal_distance_reproduces_unstable_pandas_sort(tmp_path, cfg):
    """The notebook uses unstable sort_values('abs_diff') before deduplication.

    Tie sign cannot be hardcoded: the full source-row ordering matters.
    Compare the real ingestion output with Pandas on shuffled, multi-batch
    timestamps containing hundreds of +/- equidistant pairs.
    """
    cfg = {**cfg, "timestamp_mode": "publisher_wall_clock", "batch_rows": 127}
    rng = np.random.default_rng(101)
    origin = pd.Timestamp("2024-02-06 00:00:00")
    values = []
    for i in range(1024):
        tick = origin + pd.Timedelta(seconds=5 * i)
        distance = pd.Timedelta(milliseconds=int(rng.integers(150, 2400)))
        values.extend([(tick - distance, float(i * 2)),
                       (tick + distance, float(i * 2 + 1))])
    raw_frame = pd.DataFrame(values, columns=["timestamp", "U1"])
    raw_frame = raw_frame.sample(frac=1, random_state=101).reset_index(drop=True)
    reference = raw_frame.copy()
    reference["nearest_5s"] = reference["timestamp"].dt.round("5s")
    reference["abs_diff"] = (reference["timestamp"] - reference["nearest_5s"]).abs()
    expected = (reference.sort_values("abs_diff")
                .drop_duplicates("nearest_5s")
                .sort_values("nearest_5s")["U1"].to_numpy())
    source = tmp_path / "2024_U1.csv"
    raw_frame["timestamp"] = raw_frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    raw_frame.rename(columns={"timestamp": "WsDateTime"}).to_csv(source, index=False)
    output = tmp_path / "U1.parquet"
    report = normalize_sensor(source, output, "Europe/Berlin", cfg)
    assert report["aligned_rows"] == 1024
    assert report["publisher_equal_distance_tie_ticks"] == 1024
    np.testing.assert_array_equal(pq.read_table(output)["value"].to_numpy(), expected)


def test_publisher_numpy_quicksort_matches_pandas_even_with_invalid():
    from greenforge_ai.ingestion.xz_to_parquet import _publisher_quicksort_tie_winners
    rng = np.random.default_rng(30)
    ticks = np.repeat(np.arange(4000, dtype=np.int64), 2)
    offsets = np.repeat(rng.integers(1, 2400, 4000), 2).astype(np.int64)
    rows = rng.permutation(len(ticks))
    tick = ticks[rows]
    abs_diff = offsets[rows]
    expected = (pd.DataFrame({"tick": tick, "dist": abs_diff, "ordinal": np.arange(len(rows))})
                .assign(dist=lambda x: pd.to_timedelta(x["dist"], unit="ns"))
                .sort_values("dist").drop_duplicates("tick")
                .set_index("tick")["ordinal"].to_dict())
    candidate_rows = list(zip(tick.tolist(), range(len(rows))))
    observed = _publisher_quicksort_tie_winners(abs_diff, candidate_rows)
    assert observed == expected


def test_utc_instant_tie_priority_remains_original(tmp_path, cfg):
    """The publisher-only tie policy must not change UTC ingestion."""
    cfg = {**cfg, "timestamp_mode": "utc_instant", "batch_rows": 1}
    raw = tmp_path / "2024_U1.csv"
    pd.DataFrame({
        "WsDateTime": [
            "2024-01-04 19:13:58.010",
            "2024-01-04 19:14:01.990",
        ],
        "U1": [237.73346, 237.75758],
    }).to_csv(raw, index=False)
    output = tmp_path / "U1.parquet"
    report = normalize_sensor(raw, output, "Europe/Berlin", cfg)
    assert report["aligned_rows"] == 1
    assert pq.read_table(output)["value"].to_pylist() == [237.73346]
