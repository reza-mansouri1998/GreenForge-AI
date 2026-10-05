"""Observed quarter-hour aggregation and leakage-aware forecasting windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from greenforge_ai.features.interaction_features import interval_values

INTERVAL_MS = 900_000


class QuarterHour:
    def __init__(self):
        self.rows = []
        self.tail_t = np.empty(0, dtype=np.int64)
        self.tail_v = np.empty(0)

    def update(self, ticks, values):
        ticks, values = np.r_[self.tail_t, ticks], np.r_[self.tail_v, values]
        stop = np.searchsorted(ticks // INTERVAL_MS, ticks[-1] // INTERVAL_MS, side="left")
        if stop:
            self._aggregate(ticks[:stop], values[:stop])
        self.tail_t, self.tail_v = ticks[stop:], values[stop:]

    def _aggregate(self, ticks, values):
        unique, starts, sizes = np.unique(
            ticks // INTERVAL_MS * INTERVAL_MS, return_index=True, return_counts=True
        )
        valid = np.isfinite(values)
        counts = np.add.reduceat(valid.astype(np.int64), starts)
        sums = np.add.reduceat(np.where(valid, values, 0.0), starts)
        for t, n, total, size in zip(unique, counts, sums, sizes):
            self.rows.append(
                (
                    int(t),
                    int(n),
                    float(total / n) if n else np.nan,
                    float(total / n) if n == 180 and size == 180 else np.nan,
                )
            )

    def finish(self):
        if len(self.tail_t):
            self._aggregate(self.tail_t, self.tail_v)
            self.tail_t = np.empty(0, dtype=np.int64)
        return pd.DataFrame(self.rows, columns=["tick_ms", "observed_samples", "observed_mean_w", "power_w"])


# Explicit point-in-time contract. No observation after issuing enters a feature.
FORECAST_DEFAULTS = {
    "task": "tomorrow_calendar_day", "timezone": "Europe/Berlin", "issue_time": "15:00",
    "interval_minutes": 15, "history_hours": 24, "weather_delay_hours": 2,
    "minimum_history_coverage": 0.5, "maximum_last_reading_age_hours": 2.0,
    "minimum_training_days": 20, "minimum_evaluation_days": 3,
}
FORECAST_NUMERIC = [
    "power_last_observed_kw", "power_previous_15m_kw", "power_history_mean_kw",
    "power_history_std_kw", "power_history_min_kw", "power_history_max_kw",
    "history_coverage", "last_reading_age_hours", "previous_hour_temperature_c",
    "power_same_time_last_known_kw", "power_week_ago_kw", "lead_hours",
    "hour_sin", "hour_cos", "week_sin", "week_cos", "month_sin", "month_cos",
    "weekend", "utc_offset_hours",
]


def forecast_config(cfg):
    supplied = cfg.get("forecasting", {})
    if set(supplied) - set(FORECAST_DEFAULTS):
        raise ValueError(f"Unknown forecasting settings: {set(supplied) - set(FORECAST_DEFAULTS)}")
    result = {**FORECAST_DEFAULTS, **supplied}
    if result["task"] != "tomorrow_calendar_day" or result["interval_minutes"] != 15:
        raise ValueError("Forecast tomorrow's calendar day at 15-minute resolution")
    if result["timezone"] != "Europe/Berlin":
        raise ValueError("This project uses the Europe/Berlin calendar")
    import re
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):(?:00|15|30|45)", result["issue_time"]):
        raise ValueError("issue_time must be HH:MM on a quarter-hour boundary")
    if int(result["issue_time"][:2]) < 3:
        raise ValueError("Use an issue time after 03:00 to avoid ambiguous clock-change hours")
    if not 1 <= result["history_hours"] <= 168 or not 1 <= result["weather_delay_hours"] <= 48:
        raise ValueError("Invalid history or weather delay")
    if not 0 < result["minimum_history_coverage"] <= 1 or not 0 < result["maximum_last_reading_age_hours"] <= 168:
        raise ValueError("Invalid history eligibility settings")
    if result["minimum_training_days"] < 3 or result["minimum_evaluation_days"] < 1:
        raise ValueError("Invalid minimum day counts")
    return result


def _power_series(frame):
    stamps = pd.DatetimeIndex(frame["WsDateTime"])
    if stamps.tz is None or str(stamps.tz) != "UTC":
        raise ValueError("Source interval starts must be UTC-aware")
    stamps = stamps.as_unit("ns")
    if stamps.hasnans or not stamps.is_unique or not stamps.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique, sorted and non-null")
    milliseconds = stamps.as_unit("ms").asi8
    if np.any(milliseconds % INTERVAL_MS) or np.any(np.diff(milliseconds) != INTERVAL_MS):
        raise ValueError("Source must retain its complete quarter-hour clock; use null values for gaps")
    values = pd.to_numeric(frame["power_kw"], errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return pd.Series(values, index=stamps)


def _calendar_reference(targets, days, zone):
    # Local-clock references: ambiguous/nonexistent reference hours stay missing.
    naive = targets.tz_convert(zone).tz_localize(None) - pd.Timedelta(days=days)
    return naive.tz_localize(zone, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def _day_features(power, external, origin, options):
    zone = options["timezone"]
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("Forecast origin must be timezone-aware")
    origin = origin.tz_convert("UTC").as_unit("ns")
    local = origin.tz_convert(zone)
    if local.strftime("%H:%M") != options["issue_time"] or origin.second or origin.microsecond:
        raise ValueError("Origin must match the configured daily issue time exactly")
    day = local.date() + pd.Timedelta(days=1)
    start = pd.Timestamp(day, tz=zone)
    end = pd.Timestamp(day + pd.Timedelta(days=1), tz=zone)
    targets = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC").as_unit("ns")
    step = pd.Timedelta(minutes=15)
    history_start = origin - pd.Timedelta(hours=options["history_hours"])
    history_clock = pd.date_range(history_start, origin, freq="15min", inclusive="left")
    history = power.reindex(history_clock)
    past = power.loc[power.index + step <= origin].dropna()
    last_time = past.index[-1] if len(past) else pd.NaT
    last_value = float(past.iloc[-1]) if len(past) else np.nan
    age = (origin - (last_time + step)).total_seconds() / 3600 if len(past) else np.nan
    coverage = float(history.notna().sum() / len(history_clock))
    eligible = bool(coverage >= options["minimum_history_coverage"] and np.isfinite(age)
                    and age <= options["maximum_last_reading_age_hours"])
    out = pd.DataFrame({"WsDateTime": targets, "Forecast_Origin": origin,
        "Forecast_Day": day, "Feature_Window_Start": history_start,
        "Target_Start": targets, "Target_End": targets + step,
        "expected_intervals": len(targets), "history_eligible": eligible,
        "power_last_observed_kw": last_value,
        "power_previous_15m_kw": power.get(origin - 2 * step, np.nan),
        "power_history_mean_kw": history.mean(), "power_history_std_kw": history.std(ddof=0),
        "power_history_min_kw": history.min(), "power_history_max_kw": history.max(),
        "history_coverage": coverage, "last_reading_age_hours": age})
    origin_ms = np.array([origin.as_unit("ms").asm8.view("i8")], dtype=np.int64)
    temperature = interval_values(external["DWD"], origin_ms,
                                  delay_ms=int(options["weather_delay_hours"] * 3_600_000))[0]
    out["previous_hour_temperature_c"] = temperature
    for days, column in [(1, "power_same_time_last_known_kw"), (7, "power_week_ago_kw")]:
        reference = _calendar_reference(targets, days, zone)
        if days == 1:
            fallback = _calendar_reference(targets, 2, zone)
            unavailable = reference.isna() | (reference + step > origin)
            reference = reference.where(~unavailable, fallback)
        available = ~reference.isna() & (reference + step <= origin)
        values = power.reindex(reference).to_numpy(dtype=float, copy=True)
        values[~available] = np.nan
        out[column] = values
    out["lead_hours"] = (targets - origin).total_seconds() / 3600
    local_target = targets.tz_convert(zone)
    hour = local_target.hour + local_target.minute / 60
    weekday = local_target.dayofweek
    out["hour_sin"], out["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    week = weekday + hour/24
    out["week_sin"], out["week_cos"] = np.sin(2*np.pi*week/7), np.cos(2*np.pi*week/7)
    month = local_target.month - 1
    out["month_sin"], out["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    out["weekday"] = weekday.astype("int8")
    out["weekend"] = (weekday >= 5).astype("int8")
    out["utc_offset_hours"] = [stamp.utcoffset().total_seconds()/3600 for stamp in local_target]
    return out


def make_forecast_frame(frame, external, origin, cfg):
    """Predictor-only tomorrow grid, reusable for inference. No target is read."""
    return _day_features(_power_series(frame), external, origin, forecast_config(cfg))


def make_training_frame(frame, external, cfg):
    """Direct pooled day-ahead rows: one daily origin, all tomorrow intervals.

    Retain null labels and feature gaps. Training filters missing labels; evaluation
    uses complete target days. Day-ahead price and tomorrow's observed weather are
    deliberately not predictive inputs without release-time/archive evidence.
    """
    options, power = forecast_config(cfg), _power_series(frame)
    if power.empty:
        raise ValueError("Empty EDA clock")
    local = power.index.tz_convert(options["timezone"])
    rows = []
    for day in pd.date_range(local[0].date(), local[-1].date(), freq="D"):
        origin = pd.Timestamp(f"{day.date()} {options['issue_time']}", tz=options["timezone"]).tz_convert("UTC")
        forecast = _day_features(power, external, origin, options)
        if forecast.Target_Start.iloc[0] < power.index[0] or forecast.Target_End.iloc[-1] > power.index[-1] + pd.Timedelta(minutes=15):
            continue
        forecast["target_power_kw"] = power.reindex(pd.DatetimeIndex(forecast.Target_Start)).to_numpy()
        forecast["target_observed"] = forecast.target_power_kw.notna()
        rows.append(forecast)
    if not rows:
        empty = _day_features(power, external,
            pd.Timestamp(f"{local[0].date()} {options['issue_time']}", tz=options["timezone"]), options).iloc[:0]
        empty["target_power_kw"] = pd.Series(dtype=float)
        empty["target_observed"] = pd.Series(dtype=bool)
        empty["split"] = pd.Series(dtype=str)
        empty.attrs["window_diagnostics"] = {"candidate_days": 0, "complete_days_by_split": {}}
        return empty
    out = pd.concat(rows, ignore_index=True)
    day_info = out.groupby("Forecast_Day", sort=True).agg(
        observed=("target_observed", "sum"), expected=("expected_intervals", "first"),
        eligible=("history_eligible", "first"), origin=("Forecast_Origin", "first"), end=("Target_End", "max"))
    informative = day_info[(day_info.observed > 0) & day_info.eligible]
    days = list(informative.index)
    labels = {day: "excluded" for day in day_info.index}
    if len(days) >= 3:
        first, second = max(1, int(.7*len(days))), max(2, int(.85*len(days)))
        second = min(second, len(days)-1)
        first = min(first, second-1)
        validation_origin, test_origin = informative.iloc[first].origin, informative.iloc[second].origin
        for index, day in enumerate(days):
            end = informative.loc[day, "end"]
            labels[day] = ("train" if index < first and end <= validation_origin else
                           "validation" if first <= index < second and end <= test_origin else
                           "test" if index >= second else "purged")
    out["split"] = out.Forecast_Day.map(labels)
    complete = day_info[(day_info.observed == day_info.expected) & day_info.eligible]
    complete_counts = {split: sum(labels[day] == split for day in complete.index)
                       for split in ("train", "validation", "test")}
    diagnostics = {
        "candidate_days": len(day_info), "history_eligible_days": int(day_info.eligible.sum()),
        "informative_days": len(informative), "complete_days": len(complete),
        "complete_days_by_split": complete_counts,
        "missing_targets": int(out.target_power_kw.isna().sum()),
        "excluded_or_purged_days": sum(label in {"excluded", "purged"} for label in labels.values()),
        "missing_by_column": {column: int(count) for column,count in out.isna().sum().items() if count},
    }
    for column in out.select_dtypes(include=["float"]):
        out[column] = out[column].astype("float32")
    out.attrs["window_diagnostics"] = diagnostics
    out.attrs["forecast_contract"] = {**options, "schema_version": "day_ahead_v1",
        "weather_policy": "origin-only historical observations with assumed conservative delay; no future observed weather",
        "price_policy": "not a predictor; keep SMARD in EDA and scheduling",
        "target_policy": "never impute; full daily grids retained"}
    return out
