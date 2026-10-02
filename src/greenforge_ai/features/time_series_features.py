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


def make_training_frame(frame, external, cfg):
    history, horizon = cfg["history_intervals"], cfg["horizon_intervals"]
    step = pd.Timedelta(minutes=15)
    origin = frame.WsDateTime + step
    p = frame.power_kw
    out = pd.DataFrame(
        {
            "WsDateTime": origin,
            "Feature_Window_Start": origin - history * step,
            "Target_Start": frame.WsDateTime + horizon * step,
            "Target_End": frame.WsDateTime + (horizon + 1) * step,
            "power_last_15m_kw": p,
            "power_previous_15m_kw": p.shift(1),
            "power_history_mean_kw": p.rolling(history, min_periods=history).mean(),
            "power_history_std_kw": p.rolling(history, min_periods=history).std(ddof=0),
            "target_power_kw": p.shift(-horizon),
        }
    )
    ticks = origin.dt.as_unit("ms").astype("int64").to_numpy()
    target_ticks = out.Target_Start.dt.as_unit("ms").astype("int64").to_numpy()
    out["previous_hour_temperature_c"] = interval_values(external["DWD"], ticks, delay_ms=3_600_000)
    out["target_day_ahead_price_eur_mwh"] = interval_values(external["SMARD"], target_ticks)
    local = out.Target_Start.dt.tz_convert("Europe/Berlin")
    hour = local.dt.hour + local.dt.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = local.dt.dayofweek.astype("int8")
    # Use actual target coverage: an annual split would leave Jan-Apr PV with no test set.
    observed = frame.loc[p.notna(), "WsDateTime"]
    if observed.empty:
        out["split"] = pd.Series(dtype="string")
        return out.iloc[:0]
    start, end = observed.iloc[0], observed.iloc[-1] + step
    first, second = start + (end - start) * 0.7, start + (end - start) * 0.85
    split = np.full(len(out), "purged", dtype=object)
    split[out.Target_End <= first] = "train"
    split[(out.Feature_Window_Start >= first) & (out.Target_End <= second)] = "validation"
    split[out.Feature_Window_Start >= second] = "test"
    out["split"] = split
    eligible = out.loc[out.split != "purged"]
    diagnostics = {
        "candidate_windows": len(eligible),
        "missing_by_column": {name: int(count) for name, count in eligible.isna().sum().items() if count},
    }
    out = eligible.dropna().reset_index(drop=True)
    out.attrs["window_diagnostics"] = diagnostics
    for column in out.select_dtypes("float"):
        out[column] = out[column].astype("float32")
    return out
