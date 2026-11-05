"""FastAPI contract tests: tiny isolated reports; no server, ETL or MLflow I/O.

Run with: python -m pytest tests/unit/test_api.py -q
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main as api

RUN_ID = "0123456789abcdef0123456789abcdef"
RUN_LABEL = f"2026-11-04_16-00-00Z_{RUN_ID[:8]}"


@pytest.fixture
def report_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Minimal *complete* reports; never touch the user's real application runs."""
    base = tmp_path / "application_runs"
    folders = {name: base / name / RUN_LABEL for name in api.STAGES}
    for folder in folders.values():
        folder.mkdir(parents=True)

    forecast = folders["energy_forecasting"]
    anomaly = folders["anomaly_detection_degradation"]
    optimize = folders["milp_prescriptive_scheduling"]

    models = [
        {"system": "TEC_48S", "selected_model": "LightGBM", "model_uri": f"models:/m-{RUN_ID}",
         "model_reload_verified": True, "postprocessing": "none_signed_tec_power", "r2": 0.7},
        {"system": "IPE_PV", "selected_model": "Ridge", "model_uri": f"models:/m-{RUN_ID}",
         "model_reload_verified": True, "postprocessing": "nonnegative_pv_only", "r2": 0.8},
    ]
    (forecast / "run_summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "systems": models}), encoding="utf-8"
    )
    (anomaly / "run_summary.json").write_text(json.dumps({
        "run_id": RUN_ID, "systems": [
            {"system": "TEC_48S", "review_intervals": 1},
            {"system": "IPE_PV", "review_intervals": 0},
        ],
    }), encoding="utf-8")
    pd.DataFrame(models).to_csv(forecast / "forecast_summary.csv", index=False)
    for system in ("TEC_48S", "IPE_PV"):
        pd.DataFrame({
            "WsDateTime": ["2024-11-08 06:00:00", "2024-11-08 06:15:00"],
            "target_time": ["2024-11-08 06:15:00", "2024-11-08 06:30:00"],
            "target": [0.5, 1.0], "value": [0.2, 0.5], "prediction": [0.4, 0.9],
        }).to_csv(forecast / f"{system}_predictions_15min.csv", index=False)
    pd.DataFrame({
        "system": ["TEC_48S", "IPE_PV"], "review_intervals": [1, 0],
        "scored_assessment_intervals": [10, 10],
    }).to_csv(anomaly / "anomaly_review_summary.csv", index=False)
    pd.DataFrame({
        "system": ["TEC_48S", "IPE_PV"],
        "quality_valid_intervals": [20, 12], "invalid_or_missing_intervals": [0, 8],
    }).to_csv(anomaly / "anomaly_data_quality.csv", index=False)
    pd.DataFrame({"system": ["TEC_48S", "IPE_PV"],
                  "start": ["2024-11-08 06:00:00", "2024-11-09 07:00:00"],
                  "intervals": [2, 1]}).to_csv(anomaly / "anomaly_review_episodes.csv", index=False)
    (optimize / "milp_summary.json").write_text(json.dumps({
        "matched_days": 1, "actual_savings_verified": False,
        "mean_milp_cost_eur": 0.04,
    }), encoding="utf-8")
    (optimize / "milp_assumptions.json").write_text(json.dumps({
        "job_power_kw": [1.2, 1.2], "job_steps": [4, 4],
    }), encoding="utf-8")
    pd.DataFrame({
        "date": ["2024-03-25", "2024-03-26"],
        "early_cost_eur": [0.1, 0.2], "milp_cost_eur": [0.04, 0.1],
    }).to_csv(optimize / "milp_daily_decisions.csv", index=False)
    monkeypatch.setattr(api, "REPORTS", base)
    clear_cache = api._load_saved_model.cache_clear
    clear_cache()
    yield base
    clear_cache()


@pytest.fixture
def client(report_store: Path) -> TestClient:
    with TestClient(api.app) as browser:
        yield browser


def _observations(*, count: int = 673, final: float | None = 1.0) -> list[dict]:
    """Exact 15-minute wall-clock sequence; small JSON input, no files needed."""
    start = datetime(2024, 11, 1, 8, 0)
    values = np.zeros(count, dtype=float)
    return [
        {"WsDateTime": (start + timedelta(minutes=15 * i)).isoformat(),
         "power_kw": final if i == count - 1 else float(values[i])}
        for i in range(count)
    ]


def _payload(count: int = 673) -> dict:
    observations = _observations(count=count)
    last = datetime.fromisoformat(observations[-1]["WsDateTime"])
    return {"observations": observations,
            "issue_time": (last + timedelta(minutes=15)).isoformat()}


def test_liveness_does_not_require_reports(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "REPORTS", Path("/path/that/does/not/exist"))
    assert client.get("/health").json() == {
        "status": "ok", "service": "greenforge-ai-api",
    }
    assert client.get("/api/v1/status").json()["ready"] is False
    assert client.get("/api/v1/overview").status_code == 503


def test_uses_matching_complete_run_only(report_store: Path) -> None:
    older = report_store / "energy_forecasting" / "2027-01-01_fake_run"
    older.mkdir()
    (older / "run_summary.json").write_text(json.dumps({"run_id": "wrong"}))
    selected = api._find_latest_complete_run()
    assert selected is not None and selected[0] == RUN_LABEL


def test_status_overview_models_have_same_run(client: TestClient) -> None:
    status = client.get("/api/v1/status")
    assert status.status_code == 200
    assert status.json()["ready"] is True
    assert status.json()["forecast_horizon_minutes"] == 15
    assert status.json()["real_time_data_verified"] is False
    assert client.get("/api/v1/overview").json()["run_id"] == RUN_ID
    models = client.get("/api/v1/models").json()
    assert models["run_id"] == RUN_ID
    assert {item["selected_model"] for item in models["models"]} == {"Ridge", "LightGBM"}


@pytest.mark.parametrize("system", ["TEC_48S", "IPE_PV"])
def test_forecast_date_filter_and_limit(client: TestClient, system: str) -> None:
    response = client.get(f"/api/v1/forecasts/{system}", params={"date": "2024-11-08", "limit": 1})
    assert response.status_code == 200
    assert response.json()["total_matching_rows"] == 2
    assert len(response.json()["rows"]) == 1
    assert response.json()["rows"][0]["target_time"] == "2024-11-08T06:30:00"


@pytest.mark.parametrize("query", [
    {"limit": 0}, {"limit": 501}, {"date": "2024-02-30"}, {"date": "2024-11-08T00:00"},
])
def test_forecast_rejects_bad_query(client: TestClient, query: dict) -> None:
    assert client.get("/api/v1/forecasts/TEC_48S", params=query).status_code == 422


def test_unknown_system_not_accepted(client: TestClient) -> None:
    assert client.get("/api/v1/forecasts/UNKNOWN").status_code == 422
    assert client.post("/api/v1/predict/UNKNOWN", json=_payload()).status_code == 422


def test_anomaly_reports_and_episode_filter(client: TestClient) -> None:
    summary = client.get("/api/v1/anomalies")
    assert summary.status_code == 200
    assert summary.json()["confirmed_fault_labels_available"] is False
    assert len(summary.json()["data_quality"]) == 2
    filtered = client.get("/api/v1/anomalies/episodes", params={"system": "TEC_48S"})
    assert filtered.status_code == 200
    assert filtered.json()["total_matching_rows"] == 1
    assert filtered.json()["rows"][0]["system"] == "TEC_48S"


def test_anomaly_monitoring_date_filter(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monitoring API filters correctly without creating large Parquet fixtures."""
    actual_table = api._table

    def tiny_monitor(path: Path) -> pd.DataFrame:
        if path.name == "TEC_48S_monitoring_15min.parquet":
            return pd.DataFrame({
                "WsDateTime": pd.to_datetime([
                    "2024-11-08 08:00", "2024-11-08 08:15", "2024-11-09 08:00",
                ]),
                "power_kw": [0.0, 0.6, None],
                "review_flag": [False, True, False],
                "advisory": ["no_model_alert", "review_observation", "measurement_unavailable"],
            })
        return actual_table(path)

    monkeypatch.setattr(api, "_table", tiny_monitor)
    response = client.get("/api/v1/anomalies/TEC_48S/monitoring", params={
        "date": "2024-11-08", "limit": 1,
    })
    assert response.status_code == 200
    assert response.json()["total_matching_rows"] == 2
    assert response.json()["rows"][0]["advisory"] == "review_observation"


def test_model_uri_restriction_never_loads_unapproved_source(
    client: TestClient, report_store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = report_store / "energy_forecasting" / RUN_LABEL / "run_summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["systems"][0]["model_uri"] = "file:///tmp/untrusted.pkl"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(api, "_load_saved_model", lambda _: pytest.fail("Unsafe model load attempted"))
    assert client.post("/api/v1/predict/TEC_48S", json=_payload()).status_code == 503


def test_optimization_is_historical_not_realized(client: TestClient) -> None:
    response = client.get("/api/v1/optimization")
    assert response.status_code == 200
    assert response.json()["realized_factory_savings"] is False
    assert response.json()["summary"]["matched_days"] == 1
    selected = client.get("/api/v1/optimization/days", params={"date": "2024-03-26"})
    assert selected.status_code == 200
    assert selected.json()["total_matching_days"] == 1
    assert selected.json()["rows"][0]["milp_cost_eur"] == pytest.approx(0.1)


def test_corrupt_required_report_fails_closed(client: TestClient, report_store: Path) -> None:
    path = report_store / "milp_prescriptive_scheduling" / RUN_LABEL / "milp_summary.json"
    path.write_text("not JSON", encoding="utf-8")
    response = client.get("/api/v1/optimization")
    assert response.status_code == 503


@pytest.mark.parametrize("change", [
    "missing_row", "out_of_order", "wrong_issue_time", "timezone", "stale_last",
])
def test_prediction_rejects_unsafe_time_series(client: TestClient, change: str) -> None:
    data = _payload(count=674 if change == "missing_row" else 673)
    if change == "missing_row":
        del data["observations"][8]
    elif change == "out_of_order":
        data["observations"][3], data["observations"][4] = (
            data["observations"][4], data["observations"][3]
        )
    elif change == "wrong_issue_time":
        data["issue_time"] = "2024-11-25T01:00:00"
    elif change == "timezone":
        data["issue_time"] += "+00:00"
    elif change == "stale_last":
        data["observations"][-1]["power_kw"] = None
    assert client.post("/api/v1/predict/TEC_48S", json=data).status_code == 422


def test_prediction_refuses_unverified_model(client: TestClient, report_store: Path) -> None:
    path = report_store / "energy_forecasting" / RUN_LABEL / "run_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["systems"][0]["model_reload_verified"] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    response = client.post("/api/v1/predict/TEC_48S", json=_payload())
    assert response.status_code == 503


class _ConstantModel:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert features.shape == (1, 30)
        return np.array([self.prediction], dtype=float)


@pytest.mark.parametrize(
    ("system", "raw", "expected"),
    [("TEC_48S", -0.2, -0.2), ("IPE_PV", -0.2, 0.0)],
)
def test_post_predict_loads_model_and_clips_only_pv(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, system: str,
    raw: float, expected: float,
) -> None:
    seen = []

    def fake_load(uri: str) -> _ConstantModel:
        seen.append(uri)
        return _ConstantModel(raw)

    monkeypatch.setattr(api, "_load_saved_model", fake_load)
    response = client.post(f"/api/v1/predict/{system}", json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction_kw"] == pytest.approx(expected)
    assert body["forecast_interval_end"] == (
        datetime.fromisoformat(body["issue_time"]) + timedelta(minutes=15)
    ).isoformat()
    assert body["real_time_ingestion_latency_verified"] is False
    assert seen == [f"models:/m-{RUN_ID}"]
