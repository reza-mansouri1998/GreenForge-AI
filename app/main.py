"""GreenForge AI — read-only historical API and guarded 15-minute inference.

Run from the project root:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

The historical endpoints use completed application reports; they do not retrain
models, rerun ETL or solve MILPs. A complete run must include all three stages.
MLflow models are loaded only for an explicit /predict request.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import re
from time import perf_counter
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


System = Literal["TEC_48S", "IPE_PV"]
PROJECT_ROOT = Path(
    os.getenv("GREENFORGE_PROJECT_ROOT") or Path(__file__).resolve().parents[1]
).expanduser().resolve()
REPORTS = PROJECT_ROOT / "reports" / "ml_short_term" / "application_runs"
STAGES = (
    "energy_forecasting",
    "anomaly_detection_degradation",
    "milp_prescriptive_scheduling",
)
MODEL_URI_PATTERN = re.compile(r"^models:/m-[a-f0-9]{32}$")
TIMESTAMP_COLUMNS = {
    "WsDateTime", "target_time", "available_at", "interval_end",
    "start", "end_exclusive",
}


def _logger() -> logging.Logger:
    logger = logging.getLogger("greenforge_ai.api")
    if not logger.handlers:
        folder = PROJECT_ROOT / "logs" / "application"
        folder.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            folder / "fastapi.log", maxBytes=5_000_000, backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG = _logger()
app = FastAPI(
    title="GreenForge AI API",
    version="0.1.0",
    description=(
        "Historical 2024 industrial power forecasting, advisory anomaly monitoring "
        "and hypothetical energy-aware MILP scheduling. Not a live factory controller."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOG.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise
    LOG.info(
        "method=%s path=%s status=%s duration_ms=%.1f",
        request.method, request.url.path, response.status_code,
        (perf_counter() - started) * 1_000,
    )
    return response


def _find_latest_complete_run() -> tuple[str, dict[str, Path]] | None:
    """Only serve reports belonging to the same completed three-stage run."""
    folders: dict[str, Path] = {stage: REPORTS / stage for stage in STAGES}
    if not all(path.is_dir() for path in folders.values()):
        return None
    common: set[str] | None = None
    for folder in folders.values():
        names = {p.name for p in folder.iterdir() if p.is_dir()}
        common = names if common is None else common & names
    for label in sorted(common or (), reverse=True):
        paths = {stage: folder / label for stage, folder in folders.items()}
        forecast = paths["energy_forecasting"]
        anomaly = paths["anomaly_detection_degradation"]
        milp = paths["milp_prescriptive_scheduling"]
        expected = (
            forecast / "run_summary.json", anomaly / "run_summary.json",
            forecast / "forecast_summary.csv", anomaly / "anomaly_review_summary.csv",
            milp / "milp_summary.json", milp / "milp_daily_decisions.csv",
        )
        if not all(p.is_file() for p in expected):
            continue
        try:
            f = json.loads((forecast / "run_summary.json").read_text(encoding="utf-8"))
            a = json.loads((anomaly / "run_summary.json").read_text(encoding="utf-8"))
            if not f.get("run_id") or f["run_id"] != a.get("run_id"):
                continue
            if not label.endswith(f["run_id"][:8]):
                continue
        except (ValueError, OSError, KeyError, TypeError):
            continue
        return label, paths
    return None


def _run() -> tuple[str, dict[str, Path]]:
    selected = _find_latest_complete_run()
    if selected is None:
        raise HTTPException(
            503,
            "No complete three-stage application run found. From the project "
            "root run: python -m greenforge_ai.pipelines.training_pipeline "
            "--stage all --compare-forecast-candidates",
        )
    return selected


def _json(path: Path) -> dict:
    if not path.is_file():
        raise HTTPException(503, f"Required application report is missing: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        LOG.error("invalid_report file=%s reason=%s", path.name, exc)
        raise HTTPException(503, f"Cannot read report: {path.name}") from exc


def _table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise HTTPException(503, f"Required application report is missing: {path.name}")
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except (OSError, ValueError, ImportError) as exc:
        LOG.error("invalid_table file=%s reason=%s", path.name, exc)
        raise HTTPException(503, f"Cannot read report: {path.name}") from exc


def _records(frame: pd.DataFrame) -> list[dict]:
    """JSON-safe rows; preserve publisher wall-clock labels (never mark as UTC)."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif column in TIMESTAMP_COLUMNS and pd.api.types.is_string_dtype(result[column]):
            result[column] = result[column].str.replace(" ", "T", n=1, regex=False)
    return json.loads(result.to_json(orient="records"))


def _date_filter(frame: pd.DataFrame, day: str | None, column: str) -> pd.DataFrame:
    if day is None:
        return frame
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise HTTPException(422, "date must be YYYY-MM-DD") from exc
    if column not in frame:
        raise HTTPException(503, f"Report does not include {column}")
    mask = frame[column].astype(str).str.slice(0, 10).eq(parsed)
    return frame.loc[mask]


@app.get("/")
def index():
    return {"service": "GreenForge AI", "documentation": "/docs", "health": "/health"}


@app.get("/health")
def health():
    """Liveness probe; successful response does not imply model readiness."""
    return {"status": "ok", "service": "greenforge-ai-api"}


@app.get("/api/v1/status")
def status():
    selected = _find_latest_complete_run()
    if selected is None:
        return {"ready": False, "mode": "historical_decision_support",
                "detail": "No matching complete application reports found"}
    label, paths = selected
    report = _json(paths["energy_forecasting"] / "run_summary.json")
    return {
        "ready": True, "mode": "historical_decision_support",
        "run_id": report["run_id"], "report_folder": label,
        "systems": ["TEC_48S", "IPE_PV"], "forecast_horizon_minutes": 15,
        "real_time_data_verified": False, "actual_savings_verified": False,
        "report_sections": list(STAGES),
    }


@app.get("/api/v1/overview")
def overview():
    label, paths = _run()
    forecasts = _json(paths["energy_forecasting"] / "run_summary.json")
    anomalies = _json(paths["anomaly_detection_degradation"] / "run_summary.json")
    milp = _json(paths["milp_prescriptive_scheduling"] / "milp_summary.json")
    return {
        "run_id": forecasts["run_id"], "report_folder": label,
        "mode": "historical_decision_support", "forecast_horizon_minutes": 15,
        "forecasting": forecasts["systems"],
        "anomaly_monitoring": anomalies["systems"],
        "historical_optimization": milp,
        "not_live_factory_control": True,
    }


@app.get("/api/v1/models")
def models():
    _, paths = _run()
    report = _json(paths["energy_forecasting"] / "run_summary.json")
    return {
        "run_id": report["run_id"], "models": report["systems"],
        "model_source": "MLflow local logged models, not registered model aliases",
    }


@app.get("/api/v1/forecasts/{system}")
def forecasts(
    system: System,
    limit: Annotated[int, Query(ge=1, le=500)] = 96,
    date: str | None = None,
):
    _, paths = _run()
    frame = _table(paths["energy_forecasting"] / f"{system}_predictions_15min.csv")
    frame = _date_filter(frame, date, "target_time")
    return {
        "system": system, "mode": "historical_held_out_test",
        "forecast_horizon_minutes": 15,
        "timestamp_semantics": "publisher_wall_clock_interval_start",
        "available_at_semantics": (
            "Stored target_time/available_at mark the predicted interval boundary; "
            "for zero-delay decision-making the source measurement is assumed "
            "available at WsDateTime + 15 minutes. Live latency is unverified."
        ),
        "total_matching_rows": int(len(frame)),
        "rows": _records(frame.tail(limit)),
    }


@app.get("/api/v1/anomalies")
def anomaly_summary():
    _, paths = _run()
    directory = paths["anomaly_detection_degradation"]
    return {
        "mode": "historical_advisory_monitoring",
        "summary": _records(_table(directory / "anomaly_review_summary.csv")),
        "data_quality": _records(_table(directory / "anomaly_data_quality.csv")),
        "confirmed_fault_labels_available": False,
    }


@app.get("/api/v1/anomalies/episodes")
def anomaly_episodes(
    system: System | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
):
    _, paths = _run()
    frame = _table(paths["anomaly_detection_degradation"] / "anomaly_review_episodes.csv")
    if system is not None:
        frame = frame.loc[frame["system"].eq(system)]
    return {"total_matching_rows": int(len(frame)), "rows": _records(frame.tail(limit))}


@app.get("/api/v1/anomalies/{system}/monitoring")
def anomaly_monitoring(
    system: System,
    limit: Annotated[int, Query(ge=1, le=500)] = 96,
    date: str | None = None,
):
    _, paths = _run()
    frame = _table(paths["anomaly_detection_degradation"] / f"{system}_monitoring_15min.parquet")
    frame = _date_filter(frame, date, "WsDateTime")
    return {
        "system": system, "mode": "historical_advisory_monitoring",
        "confirmed_faults": None,
        "total_matching_rows": int(len(frame)), "rows": _records(frame.tail(limit)),
    }


@app.get("/api/v1/optimization")
def optimization():
    _, paths = _run()
    folder = paths["milp_prescriptive_scheduling"]
    return {
        "mode": "historical_hypothetical_two_job_milp",
        "summary": _json(folder / "milp_summary.json"),
        "assumptions": _json(folder / "milp_assumptions.json"),
        "realized_factory_savings": False,
    }


@app.get("/api/v1/optimization/days")
def optimization_days(date: str | None = None):
    _, paths = _run()
    frame = _table(paths["milp_prescriptive_scheduling"] / "milp_daily_decisions.csv")
    frame = _date_filter(frame, date, "date")
    return {
        "mode": "historical_hypothetical_two_job_milp",
        "total_matching_days": int(len(frame)), "rows": _records(frame),
    }


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    WsDateTime: datetime
    power_kw: float | None = Field(default=None, allow_inf_nan=False)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issue_time: datetime = Field(
        description="Wall-clock decision time: last WsDateTime + 15 minutes"
    )
    observations: list[Observation] = Field(
        min_length=673, max_length=2000,
        description="Consecutive 15-minute observations, oldest to newest, "
                    "covering at least 672 historical lags",
    )


@lru_cache(maxsize=4)
def _load_saved_model(model_uri: str):
    """Load only a URI retrieved from our own completed local report."""
    if not MODEL_URI_PATTERN.fullmatch(model_uri):
        raise ValueError("The model report contains an unexpected model URI")
    db = PROJECT_ROOT / "mlruns" / "tracking.db"
    if not db.is_file():
        raise FileNotFoundError("MLflow tracking database mlruns/tracking.db is missing")
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri("sqlite:///" + db.resolve().as_posix())
    return mlflow.sklearn.load_model(model_uri)


@app.post("/api/v1/predict/{system}")
def predict(system: System, payload: PredictRequest):
    """A guarded as-of replay, not an assertion of live telemetry availability."""
    if payload.issue_time.tzinfo is not None:
        raise HTTPException(422, "Use naive publisher wall-clock issue_time, not UTC")
    timestamps = [x.WsDateTime for x in payload.observations]
    if any(t.tzinfo is not None for t in timestamps):
        raise HTTPException(422, "Use naive publisher wall-clock WsDateTime values")
    if any(b - a != timedelta(minutes=15) for a, b in zip(timestamps, timestamps[1:])):
        raise HTTPException(422, "Observations must be sorted on a complete 15-minute grid")
    expected_issue = timestamps[-1] + timedelta(minutes=15)
    if payload.issue_time != expected_issue:
        raise HTTPException(
            422,
            "Decision time must equal the final measurement start + 15 minutes. "
            "Future observations and stale observations cannot be used.",
        )
    last_value = payload.observations[-1].power_kw
    if last_value is None or not np.isfinite(last_value):
        raise HTTPException(422, "The final measured power must be finite and present")

    _, paths = _run()
    models_report = _json(paths["energy_forecasting"] / "run_summary.json")
    entry = next((m for m in models_report["systems"] if m["system"] == system), None)
    if entry is None or not entry.get("model_reload_verified"):
        raise HTTPException(503, f"A verified logged MLflow model is unavailable for {system}")
    model_uri = entry.get("model_uri", "")
    if not MODEL_URI_PATTERN.fullmatch(model_uri):
        raise HTTPException(503, "This run does not have a valid logged model URI")

    try:
        from greenforge_ai.training.train_forecasting import predict_next
        model = _load_saved_model(model_uri)
        frame = pd.DataFrame({
            "WsDateTime": pd.to_datetime(timestamps),
            "power_kw": [obs.power_kw for obs in payload.observations],
        })
        output = predict_next(model, frame, system=system)
    except (ImportError, OSError, FileNotFoundError, ValueError, RuntimeError) as exc:
        LOG.exception("model_inference_unavailable system=%s", system)
        raise HTTPException(503, f"Saved model inference unavailable: {type(exc).__name__}") from exc

    return {
        "system": system, "mode": "as_of_replay_not_verified_live",
        "model_name": entry["selected_model"], "model_uri": model_uri,
        "source_interval_start": timestamps[-1].isoformat(),
        "issue_time": payload.issue_time.isoformat(),
        "forecast_interval_start": payload.issue_time.isoformat(),
        "forecast_interval_end": (payload.issue_time + timedelta(minutes=15)).isoformat(),
        "prediction_kw": float(output["prediction_kw"]),
        "postprocessing": entry.get("postprocessing"),
        "assumed_ingestion_delay_minutes": 0,
        "real_time_ingestion_latency_verified": False,
    }
