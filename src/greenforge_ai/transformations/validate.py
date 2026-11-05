"""Publisher-informed unit rules, limits and electrical consistency checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

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
    reject_zero: bool = False
    reject_float32_subnormal: bool = False

    def clean(self, values):
        with np.errstate(over="ignore", invalid="ignore"):
            result = values.astype(float, copy=True) * self.factor
        bad = np.isfinite(values) & (~np.isfinite(result) | (np.abs(result) > np.finfo(np.float32).max))
        if self.lower is not None:
            bad |= np.isfinite(result) & (result < self.lower)
        if self.upper is not None:
            bad |= np.isfinite(result) & (result > self.upper)
        if self.reject_zero:
            bad |= np.isfinite(result) & (result == 0.0)
        if self.reject_float32_subnormal:
            bad |= np.isfinite(result) & (np.abs(result) < np.finfo(np.float32).tiny)
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


def publisher_rule_for(name, system):
    """Only documented transformations, not inferred equipment-nameplate clipping.

    Unknown publisher operations are deliberately marked for review. The
    publisher's outlier checks (after formatting and rescaling) explicitly
    use 207.85..254.03 V and 360..440 V for most equipment.
    """
    if system["input_profile"] == "spark_cleaned":
        # Already publisher-cleaned: never clean a second time.
        return Rule(None, status="official_cleaned_passthrough")
    factor, status, unit = 1.0, "publisher_unverified", None
    reject_zero = False
    reject_float32_subnormal = False
    if re.fullmatch(r"THD_I[123N]|I[123N]_h(?:[2-5]|7|9|11|13|15|17|19|21|23|25|27|29|31)", name):
        # TEC_48S 2024 parity diagnostics: aligned raw I1_h2=56 and
        # THD_I1=430 correspond to cleaned publisher values 0.56 and 4.3.
        # Full-year comparisons confirm an additional tenfold reduction for
        # those channels. Apply the conversion ONCE to the aligned raw value;
        # audit other harmonic members before claiming whole-family parity.
        if system["id"] == "TEC_48S":
            factor, status, unit = 0.01, "tec_2024_current_harmonics_div100_calibrated", "%"
        else:
            factor, status, unit = 0.1, "publisher_current_harmonics_div10", "%"
    if system["id"].startswith("TEC_") and re.fullmatch(r"THD_U(?:[123]|12|23|31)|U(?:[123]|12|23|31)_h[2-5]", name):
        factor, status, unit = 0.1, "publisher_tec_voltage_harmonics_div10", "%"
    if system["id"] == "TEC_48S" and name in {
        "U1_f", "U2_f", "U3_f", "U12_f", "U23_f", "U31_f",
        "U_line_avg_f", "U_phase_avg_f",
    }:
        reject_zero = True
        status = "publisher_tec_voltage_fundamental_zero_missing"
    if system["id"] == "TEC_48S" and name in {"U_line_avg", "U_phase_avg"}:
        # Remaining publisher-null values are zero or float32 subnormals.
        reject_float32_subnormal = True
        status = "publisher_tec_voltage_average_subnormal_missing"
    if (
        system["id"] == "TEC_48S"
        and system.get("input_profile") == "spark_raw"
        and system.get("publisher_reference_year") == 2024
        and name == "IN_h3"
    ):
        # Complete 2024 TEC_48S parity evidence shows publisher-null values
        # are exactly the strictly-above-15 percent population. This is an
        # empirically inferred compatibility rule, not a universal limit.
        status = "publisher_tec_2024_in_h3_upper_bound_inferred"
    if system["id"] in {"EPI_PV", "IPE_PV"} and name == "AC_ActivePower":
        factor, status, unit = 1000., "publisher_pv_kw_to_w", "W"
    if name == "DailyYield" and system["id"] in {"EPI_PV", "IPE_PV"}:
        status, unit = "publisher_daily_yield_wh", "Wh"
    if name == "P_total" and system["id"].startswith("TEC_"):
        # Publisher's TEC meter catalog specifies W; EPI machines use other units.
        status, unit = "publisher_tec_active_power_w", "W"
    if name in REFERENCE_CHANNELS:
        return Rule("%", status="publisher_excluded_reference")
    lower = upper = None
    if (
        system["id"] == "TEC_48S"
        and system.get("input_profile") == "spark_raw"
        and system.get("publisher_reference_year") == 2024
        and name == "IN_h3"
    ):
        upper = 15.0
    elif re.fullmatch(r"U[123]", name):
        lower, upper, unit = (207., 253., "V") if system["id"] in {
            "EPI_PickAndPlace", "EPI_ScreenPrinter"} else (207.85, 254.03, "V")
        status = "publisher_voltage_outlier_band"
    elif re.fullmatch(r"U(?:12|23|31)", name):
        lower, upper, unit = (358.53, 438.21, "V") if system["id"] in {
            "EPI_PickAndPlace", "EPI_ScreenPrinter"} else (360., 440., "V")
        status = "publisher_voltage_outlier_band"
    elif name in {"Freq", "Freq_f"}:
        # Publisher 13_Outlier_Detection.LIMITS_BY_RANGE: [49, 51] Hz.
        lower, upper, unit, status = 49., 51., "Hz", "publisher_frequency_outlier_band"
    elif re.fullmatch(r"LoadType[123](?:_f)?", name):
        lower, upper, unit, status = -1., 1., "dimensionless", "publisher_load_type_bounds"
    elif re.fullmatch(r"Angle_(?:U[123]|UI[123])(?:_f)?", name):
        lower, upper, unit, status = -180., 180., "degree", "publisher_angle_bounds"
    elif name == "OpHours":
        # Notebook shows these general outlier thresholds, but coverage varies.
        lower, upper, unit, status = 2000., 25000., "h", "publisher_operating_hours_outlier_band"
    if name in system.get("scale_factors", {}):
        factor = float(system["scale_factors"][name])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"Invalid scale for {name}")
        status = "user_override_requires_reference_check"
    if name in system.get("limits", {}):
        lower, upper = system["limits"][name]
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Reversed limits for {name}")
        status = "user_override_requires_reference_check"
    return Rule(unit, factor, lower, upper, status, reject_zero, reject_float32_subnormal)


def publisher_availability_end_ms(system, cfg, names):
    """Return dataset-specific last-valid grid ticks for active availability rules.

    These are *empirical 2024 reference-availability constraints*, not
    electrical outlier limits. The source files remain untouched and the
    policy is inactive for other years, UTC-instant mode and other systems.
    """
    policy = system.get("publisher_availability")
    if policy is None:
        return {}
    if not isinstance(policy, dict) or policy.get("policy_version") != 1:
        raise ValueError("Unsupported publisher availability policy")
    if not isinstance(policy.get("reference_year"), int):
        raise ValueError("publisher_availability.reference_year must be an integer")
    if policy.get("timestamp_mode") != "publisher_wall_clock":
        raise ValueError("Publisher availability timestamps must use wall-clock mode")
    if policy.get("input_profile") != "spark_raw":
        raise ValueError("Publisher availability policy must specify spark_raw")
    ends = policy.get("end_inclusive")
    if not isinstance(ends, dict) or not ends:
        raise ValueError("Publisher availability end_inclusive must be a nonempty mapping")
    if (cfg.get("timestamp_mode") != "publisher_wall_clock"
            or cfg.get("year") != policy["reference_year"]):
        return {}
    if system["id"] != "TEC_48S" or system["input_profile"] != "spark_raw":
        raise ValueError("Publisher availability policy is scoped to TEC_48S raw data")
    if set(ends) - set(names):
        raise ValueError(f"Availability rule names missing channels: {set(ends) - set(names)}")
    epoch = datetime(1970, 1, 1)
    result = {}
    for name, raw_end in ends.items():
        if not isinstance(raw_end, str):
            raise ValueError(f"Availability timestamp for {name} must be a string")
        try:
            dt = datetime.fromisoformat(raw_end)
        except ValueError as exc:
            raise ValueError(f"Invalid availability timestamp for {name}") from exc
        if dt.tzinfo is not None or dt.year != policy["reference_year"]:
            raise ValueError(f"Availability timestamp for {name} must be a naive date in the reference year")
        offset = dt - epoch
        if offset % timedelta(seconds=5) != timedelta(0):
            raise ValueError(f"Availability timestamp for {name} must align to the five-second grid")
        result[name] = offset // timedelta(milliseconds=1)
    return result


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



def publisher_schema_plan(paths, schema_file):
    """Plan a *local* publisher-compatible TEC output, never reading references.

    Source folders remain untouched. Refuse to publish when any alias is
    uncertain or a retained input is missing. Only call this in publisher
    wall-clock mode with an explicitly enabled publisher_schema_file.
    """
    import json
    from pathlib import Path

    spec = json.loads(Path(schema_file).read_text(encoding="utf-8"))
    retained = spec.get("retained_columns")
    aliases = spec.get("source_aliases")
    if (spec.get("schema_version") != 1 or spec.get("system") != "TEC_48S"
            or spec.get("measurement_count") != 124
            or not isinstance(retained, list) or len(retained) != 121
            or len(retained) != len(set(retained))
            or not isinstance(aliases, dict)
            or set(aliases) != {"PF_total", "Q_total", "S_total"}
            or (set(aliases) & set(retained))):
        raise ValueError("Invalid TEC publisher schema; expected 121 common + 3 mapped")
    unresolved = sorted(k for k, v in aliases.items() if not isinstance(v, str) or not v)
    if unresolved:
        raise ValueError(f"Verify publisher source mapping before enabling TEC projection: {unresolved}")
    missing = sorted((set(retained) | set(aliases.values())) - set(paths))
    if missing:
        raise ValueError(f"Publisher projection missing source channels: {missing}")
    chosen = {name: paths[name] for name in retained}
    chosen.update({name: paths[source] for name, source in aliases.items()})
    excluded = {name: "Source preserved; omitted from publisher-compatible schema"
                for name in paths if name not in retained and name not in aliases.values()}
    if set(chosen) & set(excluded):
        raise AssertionError("A retained target cannot be excluded")
    return chosen, excluded, aliases.copy()
