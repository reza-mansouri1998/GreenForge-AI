"""Small deterministic MILP/constraint tests; no ETL reads or historical solves."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from greenforge_ai.optimization.constraints import ScheduleRules, validate_schedule
from greenforge_ai.optimization.scheduler import (
    _monitor_context, candidate_jobs, solve_two_jobs,
)


@pytest.fixture
def rules() -> ScheduleRules:
    return ScheduleRules()


@pytest.fixture
def workday() -> list[datetime]:
    start = datetime(2024, 4, 1, 8, 0)  # Monday
    return [start + timedelta(minutes=15 * i) for i in range(40)]


def test_default_rules_are_historical_two_job_contract(rules: ScheduleRules) -> None:
    assert rules.slots == 40
    assert rules.slot_hours == pytest.approx(0.25)
    assert rules.job_steps == (4, 4)
    assert rules.scenario_weights == pytest.approx((0.15, 0.35, 0.5))


@pytest.mark.parametrize("override", [
    {"slot_minutes": 30},
    {"job_steps": (40, 2)},
    {"grid_capacity_kw": 1.0},
    {"scenario_weights": (0.2, 0.2, 0.2)},
    {"minimum_pv_coverage": 1.1},
    {"blocked_starts": ("25:00",)},
    {"allowed_weekdays": (0, 7)},
])
def test_rejects_invalid_scheduling_assumptions(override: dict) -> None:
    with pytest.raises(ValueError):
        ScheduleRules(**override)


def test_valid_schedule_requires_consecutive_jobs(rules: ScheduleRules) -> None:
    first = datetime(2024, 4, 1, 8, 0)
    second = first + timedelta(hours=1)
    validate_schedule((first, second), rules)


@pytest.mark.parametrize("first,second", [
    ("08:00", "08:45"),  # overlap
    ("08:15", "09:30"),  # idle time
    ("08:10", "09:10"),  # off grid
    ("17:00", "18:00"),  # beyond work window
])
def test_invalid_schedule_does_not_pass(
    first: str, second: str, rules: ScheduleRules,
) -> None:
    a = datetime.fromisoformat(f"2024-04-01T{first}")
    b = datetime.fromisoformat(f"2024-04-01T{second}")
    with pytest.raises(ValueError):
        validate_schedule((a, b), rules)


def test_weekend_and_blocked_start_rejected() -> None:
    saturday = datetime(2024, 4, 6, 8, 0)
    with pytest.raises(ValueError, match="allowed"):
        validate_schedule((saturday, saturday + timedelta(hours=1)), ScheduleRules())
    monday = datetime(2024, 4, 1, 8, 0)
    with pytest.raises(ValueError, match="Blocked"):
        validate_schedule((monday, monday + timedelta(hours=1)),
                          ScheduleRules(blocked_starts=("08:00",)))


def test_candidate_costs_and_shared_pv(workday: list[datetime], rules: ScheduleRules) -> None:
    prices = np.full(rules.slots, 100.0)  # EUR/MWh
    solar = np.zeros(rules.slots)
    first = candidate_jobs(workday, prices, solar, power_kw=1.2, duration=4, rules=rules)
    assert len(first) == 37
    assert first.loc[0, "clock"] == "08:00"
    assert first.loc[0, "grid_kwh"] == pytest.approx(1.2)  # 1.2 kW * 1 h
    assert first.loc[0, "cost_eur"] == pytest.approx(0.12)
    with_solar = candidate_jobs(
        workday, prices, np.full(rules.slots, 2.0),
        power_kw=1.2, duration=4, rules=rules,
    )
    assert with_solar.loc[0, "grid_kwh"] == pytest.approx(0.0)
    assert with_solar.loc[0, "cost_eur"] == pytest.approx(0.0)


def test_candidate_generation_rejects_missing_inputs(
    workday: list[datetime], rules: ScheduleRules,
) -> None:
    price = np.full(rules.slots, 100.0)
    solar = np.zeros(rules.slots)
    solar[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        candidate_jobs(workday, price, solar, power_kw=1.2, duration=4, rules=rules)
    with pytest.raises(ValueError, match="complete"):
        candidate_jobs(workday[:-1], price, np.zeros(40), power_kw=1.2, duration=4, rules=rules)


def test_solver_chooses_valid_lowest_cost_consecutive_pair(
    workday: list[datetime], rules: ScheduleRules,
) -> None:
    # Only eight low-price slots (10:00–12:00): feasible chosen start is 10:00.
    price = np.full(40, 100.0)
    price[8:16] = 0.0
    zero_solar = np.zeros(40)
    option1, option2 = (
        candidate_jobs(workday, price, zero_solar, power_kw=power, duration=steps, rules=rules)
        for power, steps in zip(rules.job_power_kw, rules.job_steps)
    )
    result = solve_two_jobs(option1, option2, rules)
    assert result is not None
    chosen, objective = result
    assert chosen[0]["clock"] == "10:00"
    assert chosen[1]["clock"] == "11:00"
    assert objective == pytest.approx(0.0, abs=1e-8)
    validate_schedule((chosen[0]["start"], chosen[1]["start"]), rules)


def test_solver_returns_none_when_no_valid_pairs(
    workday: list[datetime], rules: ScheduleRules,
) -> None:
    price = np.full(40, 100.0)
    solar = np.zeros(40)
    first = candidate_jobs(workday, price, solar, power_kw=1.2, duration=4, rules=rules)
    second = first.copy()
    first["feasible"] = False
    assert solve_two_jobs(first, second, rules) is None


def test_monitoring_context_ignores_future_and_stale_evidence() -> None:
    issue = datetime(2024, 4, 1, 8, 0)
    frame = pd.DataFrame({
        "WsDateTime": [issue - timedelta(minutes=30), issue - timedelta(minutes=15)],
        "interval_end": [issue - timedelta(minutes=15), issue],
        "available_at": [issue - timedelta(minutes=15), issue + timedelta(minutes=1)],
        "period": ["assessment", "assessment"],
        "model_status": ["scored", "scored"],
        "review_flag": [True, True],
    })
    context = _monitor_context(frame, "TEC_48S", issue)
    assert context["monitoring_status"] == "scored"
    assert context["reviews_last_24h"] == 1  # Future alert is excluded.
    stale = frame.iloc[:1].copy()
    stale["available_at"] = issue - timedelta(hours=3)
    stale["interval_end"] = issue - timedelta(hours=3)
    assert _monitor_context(stale, "TEC_48S", issue)["monitoring_status"] == "stale"
    assert _monitor_context(None, "TEC_48S", issue)["reviews_last_24h"] is None
