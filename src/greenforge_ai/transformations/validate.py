"""Publisher-informed unit rules, limits and electrical consistency checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

# Publisher 13_Rescaling.ipynb removes these relative fundamental-reference
# channels. Keep raw/extracted files; exclude them only from analytical tables.
REFERENCE_CHANNELS = frozenset({
    "I1_fund", "I2_fund", "I3_fund", "IN_fund",
    "U1_fund", "U2_fund", "U3_fund", "U12_fund", "U23_fund", "U31_fund",
})


# The publisher's analytical PV catalog retains AC power and daily yield.
# Raw auxiliary channels are not trustworthy: coincident source samples put
# ~400 in GridFreq and ~50 in DC_Voltage_1, alongside invalid register values.
PV_REVIEW_CHANNELS = frozenset({
    "DC_Current_1", "DC_Current_2", "DC_Voltage_1", "DC_Voltage_2",
    "DC_Power_1", "DC_Power_2", "GridFreq", "TotalYield", "Systemtime",
})


def inconsistent_power(total, phases):
    """Reject a contradictory total; never choose which phase to reconstruct.

    Preserve the existing 20% relative tolerance, with a 1 W absolute floor
    so near-zero float noise does not count as a physical contradiction.
    Missing phases provide no evidence and cannot invalidate a valid total.
    """
    summed = sum(phases)
    valid = np.isfinite(total) & np.isfinite(summed)
    tolerance = np.maximum(1.0, 0.2 * np.maximum(abs(total), abs(summed)))
    return valid & (abs(total - summed) > tolerance)


@dataclass(frozen=True)
class Rule:
    unit: str | None
    factor: float = 1.0
    lower: float | None = None
    upper: float | None = None
    status: str = "documented"

    def clean(self, values):
        with np.errstate(over="ignore", invalid="ignore"):
            result = values.astype(float, copy=True) * self.factor
        bad = np.isfinite(values) & (~np.isfinite(result) | (np.abs(result) > np.finfo(np.float32).max))
        if self.lower is not None:
            bad |= np.isfinite(result) & (result < self.lower)
        if self.upper is not None:
            bad |= np.isfinite(result) & (result > self.upper)
        result[bad | ~np.isfinite(result)] = np.nan
        return result, bad

    def metadata(self):
        return asdict(self)


def rule_for(name, system):
    profile = system["input_profile"]
    if profile not in {"spark_raw", "spark_cleaned"}:
        raise ValueError("input_profile must be spark_raw or spark_cleaned")
    current = system["rated_current_a"] * system["capacity_margin"]
    power = np.sqrt(3) * system["line_voltage_v"] * current
    factor, status = 1.0, "documented"
    if profile == "spark_raw" and system["kind"] == "machine":
        if re.fullmatch(r"THD_I[123N]|I[123N]_h(?:[2-5]|7|9|11|13|15|17|19|21|23|25|27|29|31)", name):
            factor, status = 0.1, "publisher_raw_current_harmonic_rule"
        if re.fullmatch(r"THD_U.*|U(?:[123]|12|23|31)_h[2-5]|I_sys", name):
            status = "unverified_scale"
        # Publisher 13_Rescaling.ipynb, rescale_U_THD, applies /10 to TEC.
        if system["id"].startswith("TEC") and re.fullmatch(
            r"THD_U(?:[123]|12|23|31)|U(?:[123]|12|23|31)_h[2-5]", name
        ):
            factor, status = 0.1, "publisher_raw_tec_voltage_harmonic_rule"
    if profile == "spark_raw" and system["kind"] == "pv":
        status = "unverified_scale"
        # Publisher 12_Validate_Physical_limits.ipynb explicitly converts this
        # PV channel kW -> W. Do not infer scales for other PV channels.
        if system["id"] in {"IPE_PV", "EPI_PV"}:
            if name == "AC_ActivePower":
                factor, status = 1000.0, "publisher_raw_pv_kw_to_w"
            elif name == "DailyYield":
                status = "publisher_daily_yield_wh"
    if name in system.get("scale_factors", {}):
        factor, status = float(system["scale_factors"][name]), "explicit_config"
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"Invalid scale factor: {name}")
    suffix = r"(?:_f|_RMS_fund)?"
    if name in REFERENCE_CHANNELS:
        rule = Rule("%", factor, status="unverified_reference_channel")
    elif re.fullmatch(r"U[123]" + suffix, name):
        rule = Rule("V", factor, 200.9, 260.0, status)
    elif re.fullmatch(r"U(?:12|23|31)" + suffix, name):
        rule = Rule("V", factor, 350.0, 450.0, status)
    elif re.fullmatch(r"I(?:[123N]|_sys)" + suffix, name):
        rule = Rule("A", factor, 0.0, current, status)
    elif re.fullmatch(r"P[123]" + suffix, name):
        rule = Rule("W", factor, -power / 3, power / 3, status)
    elif re.fullmatch(r"P_total" + suffix, name):
        rule = Rule("W", factor, -power, power, status)
    elif re.fullmatch(r"[QS](?:[123]|_total(?:_vec|_arith)?)" + suffix, name):
        cap = power if "total" in name else power / 3
        rule = Rule(
            "var" if name.startswith("Q") else "VA",
            factor,
            -cap if name.startswith("Q") else 0.0,
            cap,
            status,
        )
    elif re.fullmatch(r"Freq(?:_f)?|GridFreq", name):
        rule = Rule("Hz", factor, 49.0, 51.0, status)
    elif name.startswith("Angle_"):
        rule = Rule("degree", factor, -180.0, 180.0, status)
    elif name.startswith(("PF", "cos_phi", "LoadType")):
        rule = Rule("dimensionless", factor, -1.0, 1.0, status)
    elif name.startswith("THD_") or re.fullmatch(r"[UI](?:[123N]|12|23|31)_h\d+", name):
        rule = Rule("%", factor, 0.0, 100.0, status)
    elif name == "AC_ActivePower":
        rule = Rule("W", factor, 0.0, power, status)
    elif name == "DailyYield":
        rule = Rule("Wh", factor, 0.0, 50000.0 if system["id"] in {"IPE_PV", "EPI_PV"} else None, status)
    elif name == "OpHours":
        rule = Rule("h", factor, 0.0, None, status)
    else:
        rule = Rule(None, factor, status="unverified_definition")
    if rule.status.startswith("unverified"):
        rule = Rule(rule.unit, factor, status=rule.status)
    bounds = system.get("limits", {}).get(name)
    if bounds is not None:
        if rule.status.startswith("unverified"):
            raise ValueError(f"Confirm {name} scale before applying limits")
        low, high = bounds
        if low is not None and high is not None and low > high:
            raise ValueError("Reversed physical limits")
        rule = Rule(rule.unit, factor, low, high, rule.status)
    return rule


class Relationships:
    """Coincident valid measurements only; do not 'repair' a failed physical equation."""

    def __init__(self):
        self.stats = {}

    def add(self, name, left, right):
        valid = np.isfinite(left) & np.isfinite(right)
        if not valid.any():
            return
        a, b = left[valid], right[valid]
        row = self.stats.setdefault(
            name, {"support": 0, "sum_left": 0.0, "sum_right": 0.0, "outside_tolerance": 0}
        )
        row["support"] += len(a)
        row["sum_left"] += float(a.sum())
        row["sum_right"] += float(b.sum())
        row["outside_tolerance"] += int(
            (abs(a - b) > np.maximum(1.0, 0.2 * np.maximum(abs(a), abs(b)))).sum()
        )

    def update(self, data):
        for prefix in ("P", "Q", "S"):
            names = [prefix + str(i) for i in (1, 2, 3)]
            if all(n in data for n in [*names, prefix + "_total"]):
                self.add(prefix + "_phase_sum", data[prefix + "_total"], sum(data[n] for n in names))
        if all(n in data for n in ("I_sys", "I1", "I2", "I3")):
            self.add("I_sys_phase_average", data["I_sys"], (data["I1"] + data["I2"] + data["I3"]) / 3)
        for suffix in ("1", "2", "3", "_total", "1_f", "2_f", "3_f", "_total_f"):
            if all(p + suffix in data for p in ("P", "Q", "S")):
                self.add(
                    "power_triangle" + suffix,
                    data["S" + suffix],
                    np.hypot(data["P" + suffix], data["Q" + suffix]),
                )

    def report(self):
        checks = {}
        for name, row in self.stats.items():
            a, b = row["sum_left"] / row["support"], row["sum_right"] / row["support"]
            checks[name] = {
                **row,
                "mean_left": a,
                "mean_right": b,
                "violation_fraction": row["outside_tolerance"] / row["support"],
                "relative_tolerance": 0.2,
                "absolute_tolerance": 1.0,
                "allowed_violation_fraction": 0.05,
                "status": "passed" if row["outside_tolerance"] / row["support"] <= 0.05 else "review_required",
            }
        return {
            "checks": checks,
            "status": "not_tested"
            if not checks
            else "review_required"
            if any(c["status"] != "passed" for c in checks.values())
            else "passed",
        }
