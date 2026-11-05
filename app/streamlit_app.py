"""GreenForge AI — historical industrial energy intelligence dashboard.

Run the existing FastAPI server first, then:
    python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501

The dashboard reads the read-only historical FastAPI endpoints. It never runs
ETL, training, anomaly fitting, MLflow model loading, or MILP optimization.
Forecast values are from the held-out 2024 test report, not live predictions.
"""

from __future__ import annotations

from datetime import date
import os
from urllib.parse import urlsplit

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests


API_DEFAULT = os.getenv("GREENFORGE_API_URL", "http://127.0.0.1:8000")
SYSTEM_NAMES = {
    "TEC_48S": "TEC_48S · Machine",
    "IPE_PV": "IPE_PV · Solar PV",
}
PAGES = (
    "Overview",
    "Forecasting",
    "Anomaly monitoring",
    "MILP scheduling",
    "Models & data quality",
)
GREEN = "#19A88A"
BLUE = "#5788CF"
AMBER = "#D69A39"
RED = "#D46372"

st.set_page_config(
    page_title="GreenForge AI | Energy intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


class APIUnavailable(RuntimeError):
    """Readable API connection, data, or HTTP error."""


def _validated_base_url(value: str) -> str:
    """Allow an explicit http(s) API endpoint, with no trailing path/query."""
    value = value.strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise APIUnavailable("Enter a complete API URL, e.g. http://127.0.0.1:8000")
    if parts.path not in {"", "/"} or parts.query or parts.fragment or parts.username or parts.password:
        raise APIUnavailable("The API URL must contain only the server address and port.")
    return value


@st.cache_data(ttl=45, show_spinner=False, max_entries=128)
def api_get(base_url: str, endpoint: str, query: tuple[tuple[str, object], ...] = ()) -> dict:
    """Fetch an API JSON document; cache per address/endpoint/parameters."""
    url = f"{base_url}{endpoint}"
    try:
        response = requests.get(url, params=dict(query), timeout=(4, 25))
        response.raise_for_status()
        document = response.json()
    except requests.exceptions.ConnectionError as exc:
        raise APIUnavailable(f"Cannot connect to FastAPI at {base_url}.") from exc
    except requests.exceptions.Timeout as exc:
        raise APIUnavailable(f"FastAPI did not respond in time for {endpoint}.") from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = str(response.json().get("detail", ""))
        except (ValueError, AttributeError):
            pass
        raise APIUnavailable(f"FastAPI returned HTTP {response.status_code} for {endpoint}. {detail}") from exc
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise APIUnavailable(f"Cannot read the API response from {endpoint}: {exc}") from exc
    if not isinstance(document, dict):
        raise APIUnavailable(f"The API returned an unexpected response for {endpoint}.")
    return document


def load(base_url: str, endpoint: str, **params: object) -> dict:
    """Streamlit-friendly wrapper: use query keys consistently for caching."""
    query = tuple(sorted((key, value) for key, value in params.items() if value is not None))
    try:
        return api_get(base_url, endpoint, query)
    except APIUnavailable as exc:
        st.error(str(exc), icon="⚠️")
        st.stop()
        raise  # For type-checkers only.


def records(data: dict, key: str = "rows") -> pd.DataFrame:
    rows = data.get(key, [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def number(value: object, decimals: int = 3, fallback: str = "—") -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if not pd.notna(result):
        return fallback
    return f"{result:,.{decimals}f}"


def fraction(value: object, decimals: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def safe_percent_change(reference: object, comparison: object) -> float | None:
    try:
        a, b = float(reference), float(comparison)
        return 100 * (a - b) / a if a > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def model_entry(overview: dict, system: str) -> dict:
    return next(
        (item for item in overview.get("forecasting", []) if item.get("system") == system), {}
    )


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(100,116,139,.15); }
        .gf-eyebrow {font-size:.76rem; font-weight:750; letter-spacing:.14em;
            text-transform:uppercase; color:#129D87; margin-bottom:0.28rem;}
        .gf-subtitle {font-size:1rem; color:#8B98A9; margin:-.15rem 0 1.3rem;}
        .gf-note {border-left:3px solid #19A88A; background:rgba(25,168,138,.07);
            border-radius:0 8px 8px 0; padding:12px 16px; margin:14px 0 22px;
            font-size:.9rem; line-height:1.5;}
        .gf-note strong { color:#129D87; }
        .gf-hero {padding:24px 28px; border-radius:16px;
            border:1px solid rgba(25,168,138,.25);
            background:linear-gradient(120deg,rgba(25,168,138,.16),rgba(52,111,166,.06));
            margin-bottom:1.1rem;}
        .gf-hero h1 {margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-.04em;}
        .gf-hero p {margin:9px 0 0; opacity:.78; font-size:1.02rem;}
        div[data-testid="stMetric"] {border:1px solid rgba(100,116,139,.18);
            border-radius:12px; padding:15px 17px; background:rgba(100,116,139,.035);}
        div[data-testid="stMetric"] label {font-weight:650;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, subtitle: str, eyebrow: str = "2024 HISTORICAL REPLAY") -> None:
    # Constant authored text only — never interpolate API-supplied values into HTML.
    st.markdown(f'<div class="gf-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def _notice() -> None:
    st.markdown(
        """<div class="gf-note"><strong>Historical decision-support demonstration.</strong>
        Forecasts are retrospective held-out evaluations; anomaly flags are advisory, not
        confirmed faults; job schedules are hypothetical, not verified factory savings.
        Live telemetry and publication latency have not been validated.</div>""",
        unsafe_allow_html=True,
    )


def _render_table(frame: pd.DataFrame, *, height: int = 340) -> None:
    if frame.empty:
        st.info("No rows match the current selection.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True, height=height)


def _download(frame: pd.DataFrame, filename: str, label: str = "Download displayed rows as CSV") -> None:
    if not frame.empty:
        st.download_button(
            label,
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
        )


def _series_chart(frame: pd.DataFrame, x: str, series: list[tuple[str, str, str]],
                  *, ylabel: str, height: int = 350, bars: bool = False) -> None:
    if frame.empty or x not in frame.columns:
        st.info("There are no chart observations for this selection.")
        return
    fig = go.Figure()
    for col, label, color in series:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if bars:
            fig.add_trace(go.Bar(x=frame[x], y=values, name=label, marker_color=color))
        else:
            fig.add_trace(go.Scatter(
                x=frame[x], y=values, name=label, mode="lines",
                line={"color": color, "width": 2.1},
                connectgaps=False, hovertemplate="%{x}<br>%{y:.4f} kW<extra>%{fullData.name}</extra>",
            ))
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.13, x=0),
        xaxis_title="", yaxis_title=ylabel,
        hovermode="x unified" if not bars else "x",
        barmode="group" if bars else "overlay",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(127,137,148,.16)", zerolinecolor="rgba(127,137,148,.23)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def overview_page(base: str, data: dict) -> None:
    st.markdown(
        """<div class="gf-hero"><div class="gf-eyebrow">ENERGY INTELLIGENCE · GREENFORGE AI</div>
        <h1>Industrial energy, clearly explained.</h1>
        <p>15-minute forecasting · anomaly review · energy-aware scheduling</p></div>""",
        unsafe_allow_html=True,
    )
    _notice()
    optimization = data.get("historical_optimization", {})
    forecast_rows = data.get("forecasting", [])
    anomaly_rows = data.get("anomaly_monitoring", [])
    flags = sum(int(row.get("review_intervals") or 0) for row in anomaly_rows)
    saving_pct = safe_percent_change(
        optimization.get("mean_early_cost_eur"), optimization.get("mean_milp_cost_eur")
    )
    a, b, c, d = st.columns(4)
    a.metric("Forecast horizon", "15 min")
    b.metric("Review-flagged intervals", f"{flags:,}")
    c.metric("Historical MILP days", str(optimization.get("matched_days", "—")))
    d.metric("Modeled mean cost saving", f"{saving_pct:.1f}%" if saving_pct is not None else "—")
    st.caption("Cost saving compares mean hypothetical-job costs with the 08:00 schedule, not factory bills.")

    st.divider()
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.subheader("Forecasting performance")
        rows = [
            {"System": SYSTEM_NAMES.get(item.get("system"), item.get("system")),
             "Selected model": item.get("selected_model"),
             "Test R²": item.get("r2"),
             "ML MAE (kW)": item.get("mae_kw"),
             "Persistence MAE (kW)": item.get("persistence_mae_kw"),
             "Saved model reload": "Verified" if item.get("model_reload_verified") else "Not verified"}
            for item in forecast_rows
        ]
        _render_table(pd.DataFrame(rows), height=160)
        st.info("Persistence currently has lower overall MAE on both systems; ML may perform better during active intervals and on RMSE.")
    with right:
        st.subheader("Anomaly review")
        anomaly_df = pd.DataFrame([
            {"System": item.get("system"), "Flagged readings": item.get("review_intervals", 0)}
            for item in anomaly_rows
        ])
        _series_chart(anomaly_df, "System", [("Flagged readings", "Review flags", GREEN)],
                      ylabel="Readings", bars=True, height=220)
        st.caption("Flags are for review. Confirmed fault labels are unavailable.")

    st.subheader("Historical optimization economics")
    economics = pd.DataFrame([
        {"Strategy": "08:00 baseline", "Mean cost (EUR)": optimization.get("mean_early_cost_eur")},
        {"Strategy": "MILP", "Mean cost (EUR)": optimization.get("mean_milp_cost_eur")},
        {"Strategy": "Hindsight · evaluation only", "Mean cost (EUR)": optimization.get("mean_hindsight_cost_eur")},
    ])
    _series_chart(economics, "Strategy", [("Mean cost (EUR)", "Mean incremental cost", BLUE)],
                  ylabel="EUR / hypothetical day", height=260, bars=True)
    audit = optimization.get("forecast_0800_availability_audit", {})
    if audit:
        st.caption(
            f"08:00 as-of check: {audit.get('08_issue_forecasts_checked', '—')} forecasts checked; "
            "assumes zero ingestion delay. Actual publication times are not verified."
        )
    st.caption(f"Selected complete pipeline run: {str(data.get('run_id', '—'))[:12]}…")


def forecasting_page(base: str, data: dict) -> None:
    _hero("15-minute energy forecasting", "Held-out 2024 predictions from the existing application reports.")
    system_col, rows_col, day_col = st.columns([1.1, 1, 1.3])
    with system_col:
        system = st.selectbox("System", list(SYSTEM_NAMES), format_func=lambda key: SYSTEM_NAMES[key])
    with rows_col:
        limit = st.slider("Maximum returned rows", min_value=24, max_value=500, value=192, step=1)
    with day_col:
        day = st.text_input("Historical target date (optional)", placeholder="YYYY-MM-DD")
    date_query = None
    if day.strip():
        try:
            date_query = date.fromisoformat(day.strip()).isoformat()
        except ValueError:
            st.warning("Date must be YYYY-MM-DD, for example 2024-04-02.")
            st.stop()
    result = load(base, f"/api/v1/forecasts/{system}", limit=limit, date=date_query)
    frame = records(result)
    model = model_entry(data, system)
    metrics = st.columns(4)
    metrics[0].metric("Selected model", model.get("selected_model", "—"))
    metrics[1].metric("Test R²", number(model.get("r2"), 3))
    metrics[2].metric("Model MAE", f"{number(model.get('mae_kw'), 4)} kW")
    metrics[3].metric("Persistence MAE", f"{number(model.get('persistence_mae_kw'), 4)} kW")
    st.caption(
        f"{result.get('total_matching_rows', 0):,} matching test intervals; showing "
        f"{len(frame):,} latest rows. Timestamps are publisher wall-clock labels, not UTC."
    )
    if frame.empty:
        st.info("No predictions on that date. Try clearing the date filter.")
        return
    frame["target_time"] = pd.to_datetime(frame.get("target_time"), errors="coerce")
    for column in ("prediction", "target", "value", "raw_prediction"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("target_time")
    st.subheader("Predicted vs measured power")
    _series_chart(
        frame, "target_time",
        [("target", "Actual", BLUE), ("prediction", "ML forecast", GREEN),
         ("value", "Last known reading", AMBER)],
        ylabel="Power · kW", height=395,
    )
    st.caption("The last-known reading is plotted at the target time for comparison; it was measured in the preceding interval.")
    if {"target", "prediction"}.issubset(frame.columns):
        frame["absolute_error_kw"] = (frame["target"] - frame["prediction"]).abs()
        st.subheader("Absolute prediction error")
        _series_chart(frame, "target_time", [("absolute_error_kw", "Absolute error", RED)],
                      ylabel="Absolute error · kW", height=235)
    if system == "IPE_PV":
        st.info("Solar forecasts are clipped to non-negative power after Ridge inference. Raw model output is retained in the table for audit.")
    st.subheader("Forecast observations")
    _render_table(frame, height=305)
    _download(frame, f"greenforge_{system.lower()}_forecasts.csv")


def anomaly_page(base: str, data: dict) -> None:
    _hero("Anomaly & degradation review", "Unsupervised operating-state monitoring; all alerts are advisory.")
    summary_data = load(base, "/api/v1/anomalies")
    summary = records(summary_data, "summary")
    quality = records(summary_data, "data_quality")
    if not summary.empty:
        cols = st.columns(2)
        for index, system in enumerate(SYSTEM_NAMES):
            subset = summary.loc[summary["system"].eq(system)]
            if subset.empty:
                continue
            item = subset.iloc[0]
            with cols[index]:
                st.markdown(f"#### {SYSTEM_NAMES[system]}")
                a, b, c = st.columns(3)
                a.metric("Review flags", str(int(item.get("review_intervals", 0))))
                b.metric("Review rate", f"{number(item.get('review_rate_pct'), 2)}%")
                c.metric("Assessed intervals", f"{int(item.get('scored_assessment_intervals', 0)):,}")
    st.caption("Review rate uses scored assessment intervals, not all 35,136 calendar slots. No confirmed-fault labels are available.")
    st.divider()

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Measurement coverage")
        if not quality.empty:
            q = quality.copy()
            q["Valid share (%)"] = (
                pd.to_numeric(q["quality_valid_intervals"], errors="coerce") /
                pd.to_numeric(q["intervals"], errors="coerce") * 100
            ).round(1)
            _render_table(q[["system", "quality_valid_intervals", "invalid_or_missing_intervals", "Valid share (%)"]], height=145)
            st.warning("Missing or invalid readings do not mean the asset was healthy. PV data has limited annual coverage.")
    with right:
        st.subheader("Review episodes")
        episode_system = st.selectbox("Episode system", ["All", *SYSTEM_NAMES.keys()],
                                      format_func=lambda key: SYSTEM_NAMES.get(key, "Both systems"))
        params = {} if episode_system == "All" else {"system": episode_system}
        episodes_data = load(base, "/api/v1/anomalies/episodes", limit=500, **params)
        episodes = records(episodes_data)
        st.metric("Reported review episodes", str(episodes_data.get("total_matching_rows", "—")))
        if not episodes.empty:
            _render_table(episodes.sort_values("start", ascending=False), height=250)
            _download(episodes, "greenforge_anomaly_episodes.csv", "Download review episodes")
        else:
            st.info("No review episodes for this selection.")

    st.divider()
    st.subheader("Monitoring timeline")
    a, b, c = st.columns([1.2, .8, 1.4])
    with a:
        system = st.selectbox("Monitoring system", list(SYSTEM_NAMES), format_func=lambda key: SYSTEM_NAMES[key])
    with b:
        limit = st.slider("Monitoring rows", min_value=24, max_value=500, value=192)
    with c:
        day = st.text_input("Monitoring date (optional)", placeholder="YYYY-MM-DD")
    date_query = None
    if day.strip():
        try:
            date_query = date.fromisoformat(day.strip()).isoformat()
        except ValueError:
            st.warning("Date must be YYYY-MM-DD.")
            st.stop()
    monitor_data = load(base, f"/api/v1/anomalies/{system}/monitoring", limit=limit, date=date_query)
    monitoring = records(monitor_data)
    st.caption(f"{monitor_data.get('total_matching_rows', 0):,} matching intervals; displaying {len(monitoring):,} rows.")
    if not monitoring.empty:
        if "WsDateTime" in monitoring:
            monitoring["WsDateTime"] = pd.to_datetime(monitoring["WsDateTime"], errors="coerce")
        if "review_flag" in monitoring:
            flags = monitoring["review_flag"].fillna(False).astype(bool).sum()
            st.caption(f"Flagged among the displayed intervals: {flags:,}. A zero count in this window does not imply no faults.")
        _render_table(monitoring, height=310)
        _download(monitoring, f"greenforge_{system.lower()}_monitoring.csv")


def optimization_page(base: str, data: dict) -> None:
    _hero("Energy-aware MILP scheduling", "Retrospective two-job scheduling using archived prices and assumed shared PV.")
    result = load(base, "/api/v1/optimization")
    summary = result.get("summary", {})
    assumptions = result.get("assumptions", {})
    days_data = load(base, "/api/v1/optimization/days")
    days = records(days_data)
    saving = safe_percent_change(summary.get("mean_early_cost_eur"), summary.get("mean_milp_cost_eur"))
    a, b, c, d = st.columns(4)
    a.metric("Historical days", str(summary.get("matched_days", "—")))
    b.metric("Mean baseline cost", f"€{number(summary.get('mean_early_cost_eur'), 4)}")
    c.metric("Mean MILP cost", f"€{number(summary.get('mean_milp_cost_eur'), 4)}")
    d.metric("Modeled mean reduction", f"{saving:.1f}%" if saving is not None else "—")
    st.warning("Not actual factory-bill savings. Jobs, grid capacity and shared PV are hypothetical; price publication timing and live telemetry are unverified.")
    if days.empty:
        st.info("There are no historical optimization days to display.")
        return
    days = days.sort_values("date")
    st.subheader("Daily electricity costs by strategy")
    _series_chart(days, "date", [
        ("early_cost_eur", "08:00 baseline", AMBER),
        ("milp_cost_eur", "MILP", GREEN),
        ("hindsight_cost_eur", "Hindsight · evaluation only", BLUE),
    ], ylabel="EUR / hypothetical day", height=340, bars=True)
    if "milp_saving_vs_early_eur" in days:
        negative = days.loc[pd.to_numeric(days["milp_saving_vs_early_eur"], errors="coerce") < 0, "date"].astype(str).tolist()
        if negative:
            st.caption(f"MILP costs more than the baseline on {len(negative)} historical day(s): {', '.join(negative)}.")
    st.subheader("Inspect a historical schedule")
    selected_day = st.selectbox("Date", days["date"].astype(str).tolist())
    row = days.loc[days["date"].astype(str).eq(selected_day)].iloc[0]
    early, optimized, hindsight = st.columns(3)
    early.metric("08:00 baseline starts", str(row.get("early_starts", "—")))
    optimized.metric("MILP starts", str(row.get("milp_starts", "—")))
    hindsight.metric("Hindsight starts", str(row.get("hindsight_starts", "—")))
    st.caption("Each scenario contains two consecutive 60-minute, 1.2 kW hypothetical jobs; hindsight is not used to choose the MILP schedule.")

    st.subheader("Historical energy and emissions")
    _series_chart(days, "date", [
        ("early_grid_kwh", "08:00 baseline", AMBER),
        ("milp_grid_kwh", "MILP", GREEN),
    ], ylabel="Grid energy · kWh / day", height=275, bars=True)
    st.metric("Mean modeled CO₂ reduction", f"{number(summary.get('mean_co2_reduction_kg'), 4)} kg/day")
    with st.expander("Scheduling assumptions and 08:00 point-in-time audit"):
        st.json(assumptions)
        if "forecast_0800_availability_audit" in summary:
            st.json(summary["forecast_0800_availability_audit"])
        st.caption("Only the 08:00–08:15 PV forecast comes from the model; 39 later slots use pretest historical profiles.")
    st.subheader("All daily decisions")
    _render_table(days, height=315)
    _download(days, "greenforge_milp_historical_days.csv")


def models_page(base: str, data: dict) -> None:
    _hero("Models & data quality", "Model provenance, evaluation contracts, and observation coverage.")
    model_data = load(base, "/api/v1/models")
    models = model_data.get("models", [])
    if not models:
        st.warning("No model records are available in this pipeline report.")
    for item in models:
        system = item.get("system")
        with st.container(border=True):
            st.subheader(SYSTEM_NAMES.get(system, str(system)))
            a, b, c, d = st.columns(4)
            a.metric("Model", str(item.get("selected_model", "—")))
            b.metric("Test R²", number(item.get("r2"), 4))
            c.metric("Test RMSE", f"{number(item.get('rmse_kw'), 4)} kW")
            d.metric("Reload verified", "Yes" if item.get("model_reload_verified") else "No")
            st.caption(f"MLflow URI: {item.get('model_uri', '—')}")
            st.caption(
                f"Test observations: {int(item.get('testing_rows') or 0):,} · "
                f"Training observations: {int(item.get('training_rows') or 0):,} · "
                f"Selection: {item.get('selection_policy', '—')}"
            )
            ma = item.get("model_mae_on_matched_kw", item.get("mae_kw"))
            pa = item.get("persistence_mae_on_matched_kw", item.get("persistence_mae_kw"))
            st.write(f"**Comparable MAE:** ML {number(ma, 4)} kW · Persistence {number(pa, 4)} kW")
            if ma is not None and pa is not None and float(ma) >= float(pa):
                st.caption("Persistence wins on overall matched MAE; active-period and RMSE results provide additional context.")
            if system == "IPE_PV":
                st.caption("API model inference must apply non-negative clipping, matching production report postprocessing.")
            with st.expander("Full model evaluation record"):
                st.json(item)

    summary = load(base, "/api/v1/anomalies")
    st.subheader("2024 observation quality")
    quality = records(summary, "data_quality")
    if not quality.empty:
        quality["Coverage (%)"] = (pd.to_numeric(quality["quality_valid_intervals"], errors="coerce") /
                                     pd.to_numeric(quality["intervals"], errors="coerce") * 100).round(2)
        _render_table(quality, height=145)
    st.subheader("Operational boundaries")
    st.markdown(
        """- **Forecast horizon:** next 15 minutes, not day-ahead.
- **Model selection:** chronological training-only cross-validation.
- **MLflow:** locally logged model URIs, without production aliases.
- **Anomaly output:** advisory reviews; no verified failure labels.
- **MILP:** archived wholesale prices, hypothetical jobs, and assumed PV sharing.
- **Decision-time contract:** historical zero-ingestion-delay assumption; live latency not established.
- **Model inference:** available through the FastAPI POST endpoint, not triggered by this dashboard.
"""
    )
    st.caption(str(model_data.get("model_source", "")))


def main() -> None:
    _style()
    with st.sidebar:
        st.markdown("## ⚡ GreenForge AI")
        st.caption("Industrial energy intelligence")
        page = st.radio("Explore", PAGES, label_visibility="collapsed")
        st.divider()
        st.markdown("**Connection**")
        api_raw = st.text_input("FastAPI URL", value=API_DEFAULT,
                                help="Local FastAPI server; do not include /docs or /api/v1.")
        if st.button("↻ Refresh report data", use_container_width=True):
            api_get.clear()
            st.rerun()
        st.caption("API data cached for 45 seconds. Dashboard is read-only.")

    try:
        base = _validated_base_url(api_raw)
        health = api_get(base, "/health")
        status = api_get(base, "/api/v1/status")
    except APIUnavailable as exc:
        st.error(str(exc), icon="⚠️")
        st.info("Start the FastAPI server in a different VS Code terminal:\n\n"
                "`python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`")
        st.stop()
        return

    with st.sidebar:
        if health.get("status") == "ok" and status.get("ready"):
            st.success("API connected · reports ready")
        else:
            st.warning("API online · reports incomplete")
        st.caption("Mode: 2024 historical replay")
        if status.get("run_id"):
            st.caption(f"Run: {str(status['run_id'])[:12]}…")

    if not status.get("ready"):
        st.warning(status.get("detail", "No complete historical report run has been found."))
        st.info("Run the complete training pipeline first, then refresh the dashboard.")
        st.stop()
        return

    overview = load(base, "/api/v1/overview")
    if page == "Overview":
        overview_page(base, overview)
    elif page == "Forecasting":
        forecasting_page(base, overview)
    elif page == "Anomaly monitoring":
        anomaly_page(base, overview)
    elif page == "MILP scheduling":
        optimization_page(base, overview)
    else:
        models_page(base, overview)

    st.divider()
    st.caption("GreenForge AI · 2024 retrospective decision support · no live factory control or verified realized savings")


if __name__ == "__main__":
    main()
