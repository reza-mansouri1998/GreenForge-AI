"""Notebook 03's forecast-informed two-job MILP and retrospective comparison.

Only one 08:00 PV prediction is genuinely 15-min ahead. Remaining 39 slots
use strictly pre-test historical PV profiles. Future measured solar is used
ONLY to score already-chosen schedules and the hindsight reference.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from greenforge_ai.features.interaction_features import artifact, project_root
from greenforge_ai.optimization.constraints import ScheduleRules, validate_schedule
from greenforge_ai.training.validation import INTERVAL, verify_predictions

MONITOR_LOOKBACK_HOURS = 24
MONITOR_MAX_AGE_MINUTES = 60


def candidate_jobs(
    times: list[datetime], prices: np.ndarray, solar: np.ndarray,
    *, power_kw: float, duration: int, rules: ScheduleRules,
) -> pd.DataFrame:
    """Build all possible starts and incremental grid-import costs in one pass."""
    n = rules.slots
    if len(times) != n or len(prices) != n or len(solar) != n:
        raise ValueError("Candidates require a complete 08:00–18:00 work grid")
    prices, solar = np.asarray(prices, dtype=float), np.asarray(solar, dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(solar).all() or (solar < 0).any():
        raise ValueError("No non-finite prices or missing/negative solar inputs are allowed")
    if duration < 1 or duration > n or power_kw <= 0:
        raise ValueError("Invalid job duration/power")
    if any(times[i + 1] - times[i] != timedelta(minutes=rules.slot_minutes) for i in range(n - 1)):
        raise ValueError("Time slots are not consecutive")
    base_solar = solar if rules.shared_pv else np.zeros_like(solar)
    baseline_grid = np.maximum(rules.other_load_kw - base_solar, 0)
    running_grid = np.maximum(rules.other_load_kw + power_kw - base_solar, 0)
    extra = np.maximum(running_grid - baseline_grid, 0) * rules.slot_hours
    cost_per_slot = extra * prices / 1000.0  # EUR/MWh → EUR/kWh
    cost_prefix = np.r_[0.0, np.cumsum(cost_per_slot)]
    grid_prefix = np.r_[0.0, np.cumsum(extra)]
    indices = np.arange(n - duration + 1, dtype=int)
    return pd.DataFrame({
        "index": indices,
        "start": [times[i] for i in indices],
        "end": [times[i] + timedelta(minutes=rules.slot_minutes * duration) for i in indices],
        "clock": [times[i].strftime("%H:%M") for i in indices],
        "feasible": [rules.other_load_kw + power_kw <= rules.grid_capacity_kw + 1e-9
                     and times[i].strftime("%H:%M") not in rules.blocked_starts
                     for i in indices],
        "cost_eur": cost_prefix[indices + duration] - cost_prefix[indices],
        "grid_kwh": grid_prefix[indices + duration] - grid_prefix[indices],
    })


def solve_two_jobs(
    first: pd.DataFrame, second: pd.DataFrame, rules: ScheduleRules,
    *, objective: str = "cost_eur",
) -> tuple[tuple[dict, dict], float] | None:
    """Solve the binary MILP, verify feasible pairs and deterministic tie-breaking."""
    n1, n2 = len(first), len(second)
    for frame in (first, second):
        required = {"index", "start", "end", "feasible", objective}
        if missing := required - set(frame):
            raise ValueError(f"Incomplete candidate table: {sorted(missing)}")
    s1 = first["index"].to_numpy(dtype=int)
    s2 = second["index"].to_numpy(dtype=int)
    valid1 = first["feasible"].to_numpy(dtype=bool)
    valid2 = second["feasible"].to_numpy(dtype=bool)
    valid_pairs = [(i, j) for i in np.flatnonzero(valid1) for j in np.flatnonzero(valid2)
                   if s2[j] == s1[i] + rules.job_steps[0]]
    if not valid_pairs:
        return None
    c = np.r_[first[objective].to_numpy(dtype=float), second[objective].to_numpy(dtype=float)]
    if not np.isfinite(c).all():
        raise ValueError("Non-finite MILP objective")
    n = n1 + n2
    choose = np.zeros((2, n))
    choose[0, :n1] = 1.0
    choose[1, n1:] = 1.0
    occupancy = np.zeros((rules.slots, n))
    for i, start in enumerate(s1):
        occupancy[start:start + rules.job_steps[0], i] = 1.0
    for j, start in enumerate(s2):
        occupancy[start:start + rules.job_steps[1], n1 + j] = 1.0
    continuity = np.r_[-s1, s2].reshape(1, -1)
    constraints = [
        LinearConstraint(choose, [1, 1], [1, 1]),
        LinearConstraint(occupancy, -np.inf, 1.0),
        LinearConstraint(continuity, rules.job_steps[0], rules.job_steps[0]),
    ]
    bounds = Bounds(np.zeros(n), np.r_[valid1, valid2].astype(float))
    integral = np.ones(n, dtype=int)
    scaled = c * 1_000_000  # Numerical stability for small EUR objectives.
    options = {"time_limit": 20.0, "mip_rel_gap": 1e-9}
    primary = milp(scaled, integrality=integral, bounds=bounds, constraints=constraints, options=options)
    if primary.status != 0 or primary.x is None:
        raise RuntimeError(f"MILP did not prove optimality: {primary.message}")
    # Secondary objective prefers the earliest start among equal-cost schedules.
    tie = milp(
        c=np.r_[s1, s2].astype(float), integrality=integral, bounds=bounds,
        constraints=constraints + [LinearConstraint(
            scaled.reshape(1, -1), -np.inf, float(primary.fun) + 0.1
        )], options=options,
    )
    selected = tie if tie.status == 0 and tie.x is not None else primary
    i, j = int(np.argmax(selected.x[:n1])), int(np.argmax(selected.x[n1:]))
    if (i, j) not in valid_pairs:
        raise ArithmeticError("MILP selected infeasible jobs")
    starts = (pd.Timestamp(first.iloc[i]["start"]).to_pydatetime(),
              pd.Timestamp(second.iloc[j]["start"]).to_pydatetime())
    validate_schedule(starts, rules)
    cost = float(c[i] + c[n1 + j])
    enumerated = min(float(c[a] + c[n1 + b]) for a, b in valid_pairs)
    if cost > enumerated + 1e-7:
        raise ArithmeticError("MILP result is worse than the enumerated feasible optimum")
    return (first.iloc[i].to_dict(), second.iloc[j].to_dict()), cost


def _selected_cost(
    options: tuple[pd.DataFrame, pd.DataFrame], starts: tuple[str, str],
    *, column: str = "cost_eur",
) -> float | None:
    picked = []
    for frame, clock in zip(options, starts):
        found = frame.loc[frame["feasible"] & frame["clock"].eq(clock)]
        if len(found) != 1:
            return None
        picked.append(found.iloc[0])
    if picked[1]["start"] != picked[0]["end"]:
        return None
    return float(sum(row[column] for row in picked))


def _monitor_context(monitor: pd.DataFrame | None, system: str, issue: datetime) -> dict[str, object]:
    result = {"system": system, "monitoring_status": "not_evaluated", "reviews_last_24h": None}
    if monitor is None or monitor.empty:
        return result
    required = {"WsDateTime", "interval_end", "available_at", "period", "model_status", "review_flag"}
    if required - set(monitor):
        raise ValueError(f"Monitoring handoff lacks {sorted(required - set(monitor))}")
    known = monitor.loc[
        (monitor["available_at"] <= issue) & (monitor["interval_end"] <= issue)
        & (monitor["available_at"] > issue - timedelta(hours=MONITOR_LOOKBACK_HOURS))
    ].sort_values("available_at")
    if known.empty:
        return result
    latest = known.iloc[-1]
    if issue - latest["available_at"] > timedelta(minutes=MONITOR_MAX_AGE_MINUTES):
        result["monitoring_status"] = "stale"
        return result
    scored = known.loc[known["period"].eq("assessment") & known["model_status"].eq("scored")]
    if scored.empty:
        return result
    result["monitoring_status"] = "scored"
    result["reviews_last_24h"] = int(scored["review_flag"].sum())
    return result


def optimize_historical(
    root: Path | str,
    pv_forecast: pd.DataFrame,
    *, rules: ScheduleRules | None = None,
    machine_monitor: pd.DataFrame | None = None,
    pv_monitor: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Backtest application-generated forecasts, never notebook CSV predictions."""
    root = project_root(root)
    rules = rules or ScheduleRules()
    verify_predictions(pv_forecast)
    # Forecast labels are used for backtest selection only, never for optimization.
    if "target" not in pv_forecast:
        raise ValueError("Historical comparison needs observed forecast targets")
    if not pv_forecast["available_at"].eq(pv_forecast["target_time"]).all():
        raise ValueError("Only zero-delay historical issue-time assumptions are supported")
    forecasts = pv_forecast.loc[pv_forecast["available_at"].dt.strftime("%H:%M").eq("08:00")].copy()
    if forecasts.empty:
        raise ValueError("No 08:00 PV forecast to evaluate")
    if rules.work_start_hour != 8:
        raise ValueError("Historical comparison is fixed to the notebook's 08:00 issue time")
    pv = pd.read_parquet(artifact("IPE_PV", "eda_15min.parquet", root))
    tec = pd.read_parquet(artifact("TEC_48S", "eda_15min.parquet", root))
    price = "DayAhead_Price_EUR_MWh"
    if price not in pv or price not in tec:
        raise ValueError("Both ETL EDA series must have archived SMARD prices")
    archive = pv[["WsDateTime", "power_kw", "coverage", price]].rename(columns={
        "power_kw": "measured_pv", price: "price_pv"
    }).merge(
        tec[["WsDateTime", price]].rename(columns={price: "price_tec"}),
        on="WsDateTime", how="outer", validate="one_to_one",
    )
    conflict = archive["price_pv"].notna() & archive["price_tec"].notna() & (
        (archive["price_pv"] - archive["price_tec"]).abs() > 0.0001
    )
    archive["price"] = archive["price_pv"].combine_first(archive["price_tec"]).mask(conflict)
    archive = archive.sort_values("WsDateTime").set_index("WsDateTime", verify_integrity=True)
    first_issue = forecasts["available_at"].min()
    historical = archive.loc[(archive.index < first_issue) & (archive["coverage"] >= rules.minimum_pv_coverage)
                             & archive["measured_pv"].notna()].copy()
    if historical.empty:
        raise ValueError("No valid historical PV observations before the first test issue")
    historical["clock"] = historical.index.strftime("%H:%M")
    grouped = historical.groupby("clock")["measured_pv"]
    prior = pd.DataFrame({
        "n": grouped.count(), "typical": grouped.median().clip(lower=0),
        "low": grouped.quantile(0.25).clip(lower=0),
    })
    prior.loc[prior["n"] < rules.minimum_prior_samples, ["typical", "low"]] = 0.0
    records = []
    skipped = {"weekend": 0, "unavailable_day": 0, "invalid_observations": 0}
    for row in forecasts.sort_values("available_at").itertuples(index=False):
        issue = row.available_at
        if issue.weekday() not in rules.allowed_weekdays:
            skipped["weekend"] += 1
            continue
        times = [issue + timedelta(minutes=rules.slot_minutes * k) for k in range(rules.slots)]
        try:
            day = archive.loc[times]
        except KeyError:
            skipped["unavailable_day"] += 1
            continue
        values = day[["measured_pv", "price", "coverage"]].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values[:, 0] < 0).any()
                or (values[:, 2] < rules.minimum_pv_coverage).any()
                or not np.isfinite(row.prediction) or not np.isfinite(row.target)
                or not np.isclose(values[0, 0], row.target, rtol=0, atol=1e-3)):
            skipped["invalid_observations"] += 1
            continue
        clocks = [t.strftime("%H:%M") for t in times]
        typical = prior.reindex(clocks)["typical"].fillna(0).to_numpy(dtype=float)
        low = prior.reindex(clocks)["low"].fillna(0).to_numpy(dtype=float)
        informed = typical.copy()
        informed[0] = max(float(row.prediction), 0.0)
        observed = values[:, 0]
        prices = values[:, 1]

        def possible(solar: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
            return tuple(candidate_jobs(
                times, prices, solar, power_kw=kw, duration=steps, rules=rules
            ) for kw, steps in zip(rules.job_power_kw, rules.job_steps))  # type: ignore[return-value]

        typical_options = possible(informed)
        low_options = possible(low)
        zero_options = possible(np.zeros(rules.slots))
        combined = []
        for j in range(2):
            objective = sum(
                weight * option[j]["cost_eur"].to_numpy(dtype=float)
                for weight, option in zip(rules.scenario_weights,
                                          (typical_options, low_options, zero_options))
            )
            expected_import = sum(
                weight * option[j]["grid_kwh"].to_numpy(dtype=float)
                for weight, option in zip(rules.scenario_weights,
                                          (typical_options, low_options, zero_options))
            )
            objective += rules.carbon_shadow_eur_per_kg * rules.emission_factor_kg_per_kwh * expected_import
            combined.append(typical_options[j].assign(objective_eur=objective))
        selected = solve_two_jobs(combined[0], combined[1], rules, objective="objective_eur")
        if selected is None:
            skipped["invalid_observations"] += 1
            continue
        chosen_starts = tuple(part["clock"] for part in selected[0])
        realized = possible(observed)
        opening = ("08:00", "09:00")
        early_cost = _selected_cost(realized, opening)
        chosen_cost = _selected_cost(realized, chosen_starts)
        early_grid = _selected_cost(realized, opening, column="grid_kwh")
        chosen_grid = _selected_cost(realized, chosen_starts, column="grid_kwh")
        hindsight = solve_two_jobs(realized[0], realized[1], rules)
        if any(x is None for x in (early_cost, chosen_cost, early_grid, chosen_grid)) or hindsight is None:
            raise ValueError("A historical strategy became infeasible under matching constraints")
        hindsight_starts = tuple(part["clock"] for part in hindsight[0])
        hindsight_cost = float(hindsight[1])
        hindsight_grid = _selected_cost(realized, hindsight_starts, column="grid_kwh")
        if hindsight_grid is None or hindsight_cost > min(early_cost, chosen_cost) + 1e-7:
            raise ArithmeticError("Hindsight must be the least-cost realized schedule")
        tec_status = _monitor_context(machine_monitor, "TEC_48S", issue)
        pv_status = _monitor_context(pv_monitor, "IPE_PV", issue)
        records.append({
            "date": issue.date(), "early_starts": " + ".join(opening),
            "milp_starts": " + ".join(chosen_starts),
            "hindsight_starts": " + ".join(hindsight_starts),
            "early_cost_eur": early_cost, "milp_cost_eur": chosen_cost,
            "hindsight_cost_eur": hindsight_cost,
            "early_grid_kwh": early_grid, "milp_grid_kwh": chosen_grid,
            "hindsight_grid_kwh": hindsight_grid,
            "estimated_milp_objective_eur": float(selected[1]),
            "tec_monitoring_status": tec_status["monitoring_status"],
            "tec_reviews_last_24h": tec_status["reviews_last_24h"],
            "pv_monitoring_status": pv_status["monitoring_status"],
            "pv_reviews_last_24h": pv_status["reviews_last_24h"],
        })
    if not records:
        raise ValueError(f"No complete matched historical workdays. Skipped: {skipped}")
    results = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    for policy in ("early", "milp", "hindsight"):
        results[f"{policy}_co2_kg"] = results[f"{policy}_grid_kwh"] * rules.emission_factor_kg_per_kwh
    results["milp_saving_vs_early_eur"] = results["early_cost_eur"] - results["milp_cost_eur"]
    results["milp_regret_eur"] = results["milp_cost_eur"] - results["hindsight_cost_eur"]
    results["grid_reduction_kwh"] = results["early_grid_kwh"] - results["milp_grid_kwh"]
    results["co2_reduction_kg"] = results["early_co2_kg"] - results["milp_co2_kg"]
    summary = {
        "matched_days": len(results), "skipped": skipped,
        "mean_early_cost_eur": float(results["early_cost_eur"].mean()),
        "mean_milp_cost_eur": float(results["milp_cost_eur"].mean()),
        "mean_hindsight_cost_eur": float(results["hindsight_cost_eur"].mean()),
        "mean_cost_saving_eur": float(results["milp_saving_vs_early_eur"].mean()),
        "mean_grid_reduction_kwh": float(results["grid_reduction_kwh"].mean()),
        "mean_co2_reduction_kg": float(results["co2_reduction_kg"].mean()),
        "hypothetical_factory_jobs": True,
        "actual_savings_verified": False,
        "full_day_pv_forecasts_available": False,
        "future_pv_used_for_selection": False,
        "pv_prediction_minutes": 15,
        "remaining_pv_slots_from_pretest_profile": rules.slots - 1,
        "monitoring_is_advisory": True,
        "electricity_prices_are_archived_wholesale": True,
        "price_publication_times_verified": False,
        "price_known_at_08_assumed": True,
    }
    return results, summary
