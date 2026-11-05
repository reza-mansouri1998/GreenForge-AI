"""Explicit, testable rules for the notebook's illustrative factory jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

import numpy as np

from greenforge_ai.features.emissions import UBA_FACTORS_KG_PER_KWH


@dataclass(frozen=True)
class ScheduleRules:
    """Research assumptions, NOT verified factory work orders or tariffs."""

    work_start_hour: int = 8
    work_end_hour: int = 18
    slot_minutes: int = 15
    job_power_kw: tuple[float, float] = (1.2, 1.2)
    job_steps: tuple[int, int] = (4, 4)
    grid_capacity_kw: float = 4.0
    other_load_kw: float = 0.0
    shared_pv: bool = True
    allowed_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    blocked_starts: tuple[str, ...] = ()
    minimum_pv_coverage: float = 0.80
    minimum_prior_samples: int = 8
    scenario_weights: tuple[float, float, float] = (0.15, 0.35, 0.50)
    carbon_shadow_eur_per_kg: float = 0.02
    emission_factor_kg_per_kwh: float = UBA_FACTORS_KG_PER_KWH[2024]

    def __post_init__(self) -> None:
        if self.slot_minutes != 15 or (self.work_end_hour - self.work_start_hour) * 60 % self.slot_minutes:
            raise ValueError("The benchmark assumes complete 15-minute workday slots")
        if not (0 <= self.work_start_hour < self.work_end_hour <= 24):
            raise ValueError("Invalid work window")
        if len(self.job_power_kw) != 2 or len(self.job_steps) != 2:
            raise ValueError("The evaluated MILP has exactly two consecutive jobs")
        if any(not isfinite(p) or p <= 0 for p in self.job_power_kw) or any(n < 1 for n in self.job_steps):
            raise ValueError("Job power and durations must be positive")
        if sum(self.job_steps) > self.slots:
            raise ValueError("Jobs cannot fit inside the working window")
        if (not isfinite(self.other_load_kw) or self.other_load_kw < 0
                or not isfinite(self.grid_capacity_kw) or self.grid_capacity_kw <= 0):
            raise ValueError("Invalid power/capacity assumptions")
        if self.other_load_kw + max(self.job_power_kw) > self.grid_capacity_kw + 1e-9:
            raise ValueError("At least one job exceeds the assumed grid limit")
        if len(self.scenario_weights) != 3 or any(not isfinite(w) or w < 0 for w in self.scenario_weights):
            raise ValueError("Scenario weights must be nonnegative")
        if not np.isclose(sum(self.scenario_weights), 1.0, atol=1e-12):
            raise ValueError("Scenario weights must total 1")
        if not 0 <= self.minimum_pv_coverage <= 1 or self.minimum_prior_samples < 1:
            raise ValueError("Invalid PV evidence thresholds")
        if (not isfinite(self.carbon_shadow_eur_per_kg) or self.carbon_shadow_eur_per_kg < 0
                or not isfinite(self.emission_factor_kg_per_kwh) or self.emission_factor_kg_per_kwh < 0):
            raise ValueError("Invalid carbon assumptions")
        if any(d not in range(7) for d in self.allowed_weekdays):
            raise ValueError("Allowed weekdays must be in 0..6")
        for clock in self.blocked_starts:
            try:
                datetime.strptime(clock, "%H:%M")
            except ValueError as exc:
                raise ValueError(f"Invalid blocked start: {clock}") from exc

    @property
    def slots(self) -> int:
        return (self.work_end_hour - self.work_start_hour) * 60 // self.slot_minutes

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60


def validate_schedule(starts: tuple[datetime, datetime], rules: ScheduleRules) -> None:
    """Reject out-of-window, overlapping, non-continuous or prohibited jobs."""
    if len(starts) != 2:
        raise ValueError("Two starts are required")
    first, second = starts
    if first.date() != second.date() or first.weekday() not in rules.allowed_weekdays:
        raise ValueError("Jobs must be on one allowed workday")
    day_start = first.replace(hour=rules.work_start_hour, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(hours=rules.work_end_hour - rules.work_start_hour)
    slot = timedelta(minutes=rules.slot_minutes)
    if any(t < day_start or t >= day_end or (t - day_start) % slot for t in starts):
        raise ValueError("Job start is outside the 15-minute work grid")
    if any(t.strftime("%H:%M") in rules.blocked_starts for t in starts):
        raise ValueError("Blocked start time")
    if second != first + slot * rules.job_steps[0]:
        raise ValueError("Jobs must be back-to-back without overlap or idle time")
    if second + slot * rules.job_steps[1] > day_end:
        raise ValueError("Second job exceeds working hours")
    if rules.other_load_kw + max(rules.job_power_kw) > rules.grid_capacity_kw + 1e-9:
        raise ValueError("Grid capacity exceeded")
