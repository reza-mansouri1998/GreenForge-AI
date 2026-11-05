"""Run GreenForge 15-minute forecasting, anomaly review and MILP with MLflow.

Run from the project root after the existing ETL has published its manifest:
    python -m greenforge_ai.pipelines.training_pipeline --stage all

Only writes application_runs and MLflow artifacts; notebook_runs stays untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

from greenforge_ai.features.interaction_features import project_root
from greenforge_ai.optimization.constraints import ScheduleRules
from greenforge_ai.optimization.scheduler import optimize_historical
from greenforge_ai.training.train_anomaly import (
    fit_anomaly, attach_forecast_residuals, review_episodes,
)
from greenforge_ai.training.train_forecasting import train_forecast
from greenforge_ai.training.validation import FORECAST_FEATURES, SYSTEMS, load_eda
from greenforge_ai.utils.logger import get_logger, setup_logging, stage
from greenforge_ai.utils.streaming import atomic_path, write_json

LOG = get_logger(__name__, channel="training")
EXPERIMENT_NAME = "GreenForge_AI_15min_Forecasting_Anomaly_MILP"


def _configure_mlflow(root: Path):
    """SQLite metadata and durable local artifacts under the existing mlruns/."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError("MLflow is required: install the corrected requirements.txt") from exc
    store = root / "mlruns"
    store.mkdir(parents=True, exist_ok=True)
    tracking = "sqlite:///" + (store / "tracking.db").resolve().as_posix()
    mlflow.set_tracking_uri(tracking)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        location = (store / "artifacts").resolve().as_uri()
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _report_dir(root: Path, section: str, report_label: str) -> Path:
    # Human-friendly UTC label + short run ID: stable, sortable and collision-safe.
    # MLflow's full run ID still lives in each run_summary.json.
    path = root / "reports" / "ml_short_term" / "application_runs" / section / report_label
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    with atomic_path(path) as temp:
        frame.to_csv(temp, index=False)


def _save_parquet(frame: pd.DataFrame, path: Path) -> None:
    with atomic_path(path) as temp:
        frame.to_parquet(temp, engine="pyarrow", compression="zstd", index=False)


def _log_numbers(mlflow, data: dict, prefix: str = "") -> None:
    for key, value in data.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
            if math.isfinite(float(value)):
                mlflow.log_metric(prefix + str(key), float(value))


def _verify_model_in_new_process(mlflow, model_uri: str, example: pd.DataFrame, expected: np.ndarray) -> None:
    """Reload a locally trained MLflow model in a separate Python process.

    Preserve the *exact* float64 feature values. pandas.DataFrame.to_json()
    rounds floats by default, and a tiny change can cross a LightGBM split.
    Loading an NPZ file with allow_pickle=False does not deserialize objects.
    """
    if example.empty:
        raise ValueError("Cannot verify a model with an empty input example")

    script = """
import json
import sys
import numpy as np
import pandas as pd
import mlflow.sklearn

with np.load(sys.argv[2], allow_pickle=False) as values:
    frame = pd.DataFrame(
        values["features"], columns=values["feature_names"].tolist()
    )
model = mlflow.sklearn.load_model(sys.argv[1])
predictions = np.asarray(model.predict(frame), dtype=np.float64)
print(json.dumps(predictions.tolist(), allow_nan=False))
"""
    environment = os.environ.copy()
    environment["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()

    with tempfile.TemporaryDirectory(prefix="greenforge_reload_") as temp_dir:
        example_file = Path(temp_dir) / "features.npz"
        np.savez(
            example_file,
            features=example.to_numpy(dtype=np.float64),
            feature_names=np.asarray(example.columns.to_list(), dtype=str),
        )
        process = subprocess.run(
            [sys.executable, "-c", script, model_uri, str(example_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Fresh-process MLflow reload failed for {model_uri}: "
            f"{process.stderr[-1800:]}"
        )
    try:
        received = np.asarray(
            json.loads(process.stdout.strip().splitlines()[-1]), dtype=np.float64
        )
    except (ValueError, IndexError, TypeError) as exc:
        raise RuntimeError("Model reload did not return numeric predictions") from exc

    np.testing.assert_allclose(
        received,
        np.asarray(expected, dtype=np.float64),
        rtol=1e-5,
        atol=1e-7,
        err_msg="Reloaded model predictions differ from fitted model predictions",
    )


def _forecast_stage(
    root: Path, run_id: str, report_label: str, mlflow, *, compare: bool, register: bool,
    verify_model_reload: bool,
):
    from mlflow.models import infer_signature

    directory = _report_dir(root, "energy_forecasting", report_label)
    summaries, results = [], {}
    for system in SYSTEMS:
        with mlflow.start_run(run_name=f"forecast_{system}", nested=True):
            with stage(f"forecast_{system}", logger=LOG):
                observations = load_eda(root, system)
                result = train_forecast(system, observations, select_by_cv=compare)
                name = f"greenforge_{system.lower()}_15min"
                # Use the exact features already built during training; avoid
                # rebuilding every lag/rolling window merely for an example.
                example = result.input_example
                signature = infer_signature(example, result.model.predict(example))
                # skops is retained for Ridge/LightGBM, with only the known
                # model-specific types allowlisted. CatBoost's native estimator
                # does not have guaranteed skops support, so store the locally
                # fitted, trusted sklearn pipeline using cloudpickle. NEVER
                # load a cloudpickle artifact from an untrusted source.
                serialization = (
                    mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                    if result.model_name == "CatBoost"
                    else mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
                )
                trusted_types = ["numpy.dtype"]
                if result.model_name == "LightGBM":
                    trusted_types.extend([
                        "collections.OrderedDict",
                        "lightgbm.basic.Booster",
                        "lightgbm.sklearn.LGBMRegressor",
                    ])
                model_info = mlflow.sklearn.log_model(
                    result.model, name=name,
                    registered_model_name=name if register else None,
                    signature=signature, input_example=example,
                    serialization_format=serialization,
                    skops_trusted_types=(
                        trusted_types if result.model_name != "CatBoost" else None
                    ),
                    metadata={"forecast_horizon_minutes": 15, "system": system,
                              "timestamp_semantics": "publisher_wall_clock_interval_start"},
                )
                if verify_model_reload:
                    _verify_model_in_new_process(
                        mlflow, model_info.model_uri, example, result.model.predict(example),
                    )
                    mlflow.log_metric("model_reload_verified", 1.0)
                mlflow.log_params({
                    "system": system, "model_name": result.model_name,
                    "forecast_horizon_minutes": 15,
                    "selection_policy": "train_only_timeseries_cv" if compare else "notebook_train_cv_preset",
                    "features": len(FORECAST_FEATURES),
                    "postprocessing": result.postprocessing,
                    "model_serialization": serialization,
                })
                mlflow.log_dict({"feature_names": list(FORECAST_FEATURES),
                                 "model_parameters": _safe_json(result.model_parameters),
                                 "selection_scores": result.selection_scores,
                                 "data_source": "ETL eda_15min.parquet",
                                 "held_out_test_used_to_select_model": False,
                                 "timestamp_semantics": "publisher_wall_clock",
                                 "assumed_ingestion_delay_minutes": 0,
                                 "postprocessing": result.postprocessing,
                                 "model_serialization": serialization,
                                 "model_reload_verified": bool(verify_model_reload)}, "forecast_contract.json")
                _log_numbers(mlflow, result.test_metrics, "test_")
                if result.baseline_metrics:
                    _log_numbers(mlflow, result.baseline_metrics, "persistence_")
                _log_numbers(mlflow, result.comparison_metrics, "comparison_")
                for candidate, score in result.selection_scores.items():
                    mlflow.log_metric(f"cv_r2_{candidate.lower()}", score)
                path = directory / f"{system}_predictions_15min.parquet"
                _save_parquet(result.predictions, path)
                _save_csv(result.predictions, path.with_suffix(".csv"))
                row = {
                    "system": system, "selected_model": result.model_name,
                    "model_uri": model_info.model_uri,
                    "training_rows": result.train_rows, "testing_rows": result.test_rows,
                    **result.test_metrics,
                    "persistence_mae_kw": (result.baseline_metrics or {}).get("mae_kw"),
                    **result.comparison_metrics,
                    "postprocessing": result.postprocessing,
                    "model_reload_verified": bool(verify_model_reload),
                    "selection_policy": "CV comparison" if compare else "Notebook training-CV preset",
                }
                summaries.append(row)
                results[system] = result.predictions
                LOG.info("FORECAST %s model=%s rows=%d", system, result.model_name, len(result.predictions))
    _save_csv(pd.DataFrame(summaries), directory / "forecast_summary.csv")
    write_json(directory / "run_summary.json", _safe_json({"run_id": run_id, "systems": summaries}))
    # Reports live only under application_runs; MLflow stores metrics and model artifacts.
    return results


def _anomaly_stage(
    root: Path, run_id: str, report_label: str, mlflow,
    forecasts: dict[str, pd.DataFrame] | None = None,
):
    directory = _report_dir(root, "anomaly_detection_degradation", report_label)
    summaries, handoffs, all_reviews = [], {}, []
    for system in SYSTEMS:
        with mlflow.start_run(run_name=f"anomaly_{system}", nested=True):
            with stage(f"anomaly_{system}", logger=LOG):
                observations = load_eda(root, system)
                result = fit_anomaly(system, observations)
                main_monitor = attach_forecast_residuals(result.monitoring, (forecasts or {}).get(system))
                _save_parquet(main_monitor, directory / f"{system}_monitoring_15min.parquet")
                models_meta = {
                    state: {"method": model.method, "setting": model.setting,
                            "features": list(model.features),
                            "reference_rows": model.reference_rows,
                            "calibration_rows": model.calibration_rows,
                            "threshold_raw": model.raw_threshold,
                            "calibration_center": model.center,
                            "calibration_scale": model.spread}
                    for state, model in result.models.items()
                }
                mlflow.log_params({"system": system, "review_share": 0.02,
                                   "minimum_coverage": 0.80, "reference_share": 0.60,
                                   "calibration_share": 0.20,
                                   "anomaly_method": "notebook_selected_regime_models"})
                _log_numbers(mlflow, result.summary)
                mlflow.log_dict(_safe_json(models_meta), "anomaly_model_contract.json")
                with tempfile.TemporaryDirectory() as scratch:
                    bundle = Path(scratch) / f"{system}_anomaly.joblib"
                    joblib.dump({"system": system, "models": result.models,
                                 "model_contract": models_meta,
                                 "training_input": "ETL eda_15min.parquet",
                                 "ingestion_delay_assumption_minutes": 0}, bundle,
                                compress=3)
                    mlflow.log_artifact(str(bundle), artifact_path="anomaly_models")
                summaries.append(result.summary)
                all_reviews.append(main_monitor)
                handoffs[system] = main_monitor
                if system == "TEC_48S":
                    # Separate, strictly pre-test machine model for Notebook 03's dates.
                    cutoff = pd.Timestamp("2024-03-24 08:00:00")
                    end = pd.Timestamp("2024-04-09 00:00:00")
                    early = fit_anomaly(system, observations, early_cutoff=cutoff, early_end=end)
                    _save_parquet(early.monitoring,
                                  directory / "machine_monitoring_early_assessment_15min.parquet")
                    with tempfile.TemporaryDirectory() as scratch:
                        bundle = Path(scratch) / "TEC_48S_early_anomaly.joblib"
                        joblib.dump({"system": system, "models": early.models,
                                     "cutoff": str(cutoff), "end": str(end)}, bundle, compress=3)
                        mlflow.log_artifact(str(bundle), artifact_path="anomaly_models")
                    write_json(directory / "early_machine_summary.json", _safe_json(early.summary))
                    handoffs["TEC_48S_early"] = early.monitoring
    combined = pd.concat(all_reviews, ignore_index=True).sort_values(["available_at", "system"])
    _save_parquet(combined, directory / "anomaly_review_15min.parquet")
    _save_csv(review_episodes(combined), directory / "anomaly_review_episodes.csv")
    _save_csv(pd.DataFrame([
        {"system": s["system"], "intervals": s["intervals"],
         "quality_valid_intervals": s["quality_valid_intervals"],
         "invalid_or_missing_intervals": s["intervals"] - s["quality_valid_intervals"]}
        for s in summaries
    ]), directory / "anomaly_data_quality.csv")
    _save_csv(pd.DataFrame(summaries), directory / "anomaly_review_summary.csv")
    write_json(directory / "run_summary.json", _safe_json({"run_id": run_id, "systems": summaries}))
    # Report files stay in application_runs; avoid duplicate MLflow artifact copies.
    return handoffs


def _audit_0800_pv_forecast_availability(pv_forecast: pd.DataFrame) -> dict[str, object]:
    """Check that every 08:00 decision uses only the 07:45 source interval.

    The publisher timestamps denote the START of a 15-minute measurement
    interval. At 08:00, the 07:45-08:00 measurement is assumed available with
    zero acquisition delay, and predicts the 08:00-08:15 interval. This is a
    *historical assumption*, not measured live-data publication latency.
    """
    required = {"WsDateTime", "target_time", "available_at", "prediction"}
    missing = required - set(pv_forecast.columns)
    if missing:
        raise ValueError(f"PV forecast handoff lacks {sorted(missing)}")
    times = {}
    for name in ("WsDateTime", "target_time", "available_at"):
        times[name] = pd.to_datetime(pv_forecast[name], errors="raise")
        if times[name].isna().any():
            raise ValueError(f"PV forecast contains missing {name}")
    issues = times["available_at"]
    chosen = issues.dt.hour.eq(8) & issues.dt.minute.eq(0)
    if not chosen.any():
        raise ValueError("No 08:00-issued PV forecasts for the MILP")
    previous_start = times["WsDateTime"].loc[chosen]
    target_start = times["target_time"].loc[chosen]
    issue_time = issues.loc[chosen]
    slot = pd.Timedelta(minutes=15)
    if not (target_start.eq(issue_time).all()
            and (target_start - previous_start).eq(slot).all()
            and previous_start.dt.hour.eq(7).all()
            and previous_start.dt.minute.eq(45).all()):
        raise ValueError(
            "Look-ahead risk: an 08:00 MILP forecast must use the 07:45 "
            "observation and target the 08:00-08:15 interval"
        )
    if target_start.duplicated().any():
        raise ValueError("Duplicate 08:00 PV forecasts")
    predicted = pd.to_numeric(pv_forecast.loc[chosen, "prediction"], errors="raise")
    if not np.isfinite(predicted.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite 08:00 PV predictions")
    return {
        "status": "historical_zero_delay_assumption_passed",
        "08_issue_forecasts_checked": int(chosen.sum()),
        "source_interval_start": "07:45",
        "source_interval_end_and_decision_time": "08:00",
        "forecasted_interval": "08:00-08:15",
        "assumed_ingestion_delay_minutes": 0,
        "real_time_publication_latency_verified": False,
    }


def _optimization_stage(
    root: Path, run_id: str, report_label: str, mlflow, pv_predictions: pd.DataFrame,
    monitors: dict[str, pd.DataFrame] | None,
) -> None:
    # Verify the one genuine decision-time PV forecast BEFORE scheduling.
    # The remaining 39 slots are pre-test historical proxy estimates.
    availability_audit = _audit_0800_pv_forecast_availability(pv_predictions)
    directory = _report_dir(root, "milp_prescriptive_scheduling", report_label)
    rules = ScheduleRules()
    with mlflow.start_run(run_name="milp_historical", nested=True):
        with stage("milp_historical", logger=LOG):
            daily, summary = optimize_historical(
                root, pv_predictions, rules=rules,
                machine_monitor=(monitors or {}).get("TEC_48S_early"),
                pv_monitor=(monitors or {}).get("IPE_PV"),
            )
            _save_csv(daily, directory / "milp_daily_decisions.csv")
            _save_parquet(daily, directory / "milp_daily_decisions.parquet")
            summary["forecast_0800_availability_audit"] = availability_audit
            write_json(directory / "milp_summary.json", _safe_json(summary))
            write_json(directory / "milp_assumptions.json", _safe_json(asdict(rules)))
            mlflow.log_params({
                "scheduler": "scipy_milp_highs", "forecast_minutes": 15,
                "work_window": "08:00-18:00", "job_powers_kw": str(rules.job_power_kw),
                "job_steps": str(rules.job_steps), "grid_capacity_kw": rules.grid_capacity_kw,
                "scenario_weights": str(rules.scenario_weights),
                "pv_39_slots": "pretest_historical_profiles",
                "hindsight_uses_future_solar_for_evaluation_only": True,
                "forecast_0800_time_check": availability_audit["status"],
                "forecast_ingestion_delay_assumed_minutes": 0,
            })
            _log_numbers(mlflow, summary)
            mlflow.log_metric(
                "forecast_0800_issues_checked",
                availability_audit["08_issue_forecasts_checked"],
            )
            # Optimization reports already exist in application_runs; no duplicate copy.
            LOG.info("MILP scored %s matched historical days", summary["matched_days"])


def run(
    root: Path | str | None = None, *, stage_name: str = "all", compare_forecasting: bool = False,
    register_models: bool = False, pv_forecast_file: Path | str | None = None,
    machine_monitor_file: Path | str | None = None, pv_monitor_file: Path | str | None = None,
    verify_model_reload: bool = True,
) -> str:
    """Single entry point for reproducible MLflow-backed application runs."""
    root = project_root(root)
    if stage_name not in {"all", "forecast", "anomaly", "optimization"}:
        raise ValueError(f"Invalid stage: {stage_name}")
    if stage_name == "optimization" and pv_forecast_file is None:
        raise ValueError("--pv-forecast-file is required when optimization runs alone")
    setup_logging(root / "logs")
    mlflow = _configure_mlflow(root)
    with mlflow.start_run(run_name=f"greenforge_{stage_name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}") as parent:
        run_id = parent.info.run_id
        report_label = f"{datetime.now(timezone.utc):%Y-%m-%d_%H-%M-%SZ}_{run_id[:8]}"
        mlflow.set_tags({"project": "GreenForge AI", "pipeline": stage_name,
                         "application_report_folder": report_label,
                         "forecast_horizon": "15 minutes", "data_year": "2024",
                         "scope": "retrospective_decision_support"})
        predictions = {}
        monitors = {}
        if stage_name in {"all", "forecast"}:
            predictions = _forecast_stage(
                root, run_id, report_label, mlflow, compare=compare_forecasting,
                register=register_models, verify_model_reload=verify_model_reload,
            )
        if stage_name in {"all", "anomaly"}:
            monitors = _anomaly_stage(root, run_id, report_label, mlflow, predictions)
        if stage_name in {"all", "optimization"}:
            if stage_name == "optimization":
                path = Path(pv_forecast_file)
                predictions["IPE_PV"] = (
                    pd.read_parquet(path) if path.suffix.lower() == ".parquet"
                    else pd.read_csv(path, parse_dates=["WsDateTime", "target_time", "available_at"])
                )
                for key, source in (("TEC_48S_early", machine_monitor_file), ("IPE_PV", pv_monitor_file)):
                    if source is not None:
                        p = Path(source)
                        monitors[key] = (pd.read_parquet(p) if p.suffix.lower() == ".parquet"
                                         else pd.read_csv(p, parse_dates=["WsDateTime", "interval_end", "available_at"]))
            _optimization_stage(root, run_id, report_label, mlflow, predictions["IPE_PV"], monitors)
        LOG.info("GreenForge run completed: mlflow_run=%s application_reports=%s", run_id, report_label)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="GreenForge 15-minute ML + MILP pipeline")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "forecast", "anomaly", "optimization"), default="all")
    parser.add_argument("--compare-forecast-candidates", action="store_true",
                        help="Compare two notebook-tuned candidates using training CV only; more CPU time")
    parser.add_argument("--register-models", action="store_true",
                        help="Also register forecasters in the local MLflow model registry")
    parser.add_argument("--skip-model-reload-test", action="store_true",
                        help="Skip the fresh-process MLflow model load test to save startup time")
    parser.add_argument("--pv-forecast-file", type=Path)
    parser.add_argument("--machine-monitor-file", type=Path)
    parser.add_argument("--pv-monitor-file", type=Path)
    args = parser.parse_args()
    print(run(
        args.project_root, stage_name=args.stage,
        compare_forecasting=args.compare_forecast_candidates,
        register_models=args.register_models,
        pv_forecast_file=args.pv_forecast_file,
        machine_monitor_file=args.machine_monitor_file,
        pv_monitor_file=args.pv_monitor_file,
        verify_model_reload=not args.skip_model_reload_test,
    ))


if __name__ == "__main__":
    main()
