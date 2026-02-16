from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _control_entries(scenario_cfg: dict) -> List[Tuple[str, str, dict]]:
    out: List[Tuple[str, str, dict]] = []
    for kind in ("levers", "shocks"):
        block = scenario_cfg.get(kind, {}) or {}
        if not isinstance(block, dict):
            continue
        for name, spec in block.items():
            if isinstance(spec, dict):
                out.append((kind, str(name), spec))
    return out


def _autofill_enabled(scenario_autofill_cfg: Optional[dict]) -> bool:
    cfg = scenario_autofill_cfg or {}
    return bool(cfg.get("enabled", False))


def _with_autofill_spec(
    control_name: str,
    spec: dict,
    scenario_autofill_cfg: Optional[dict],
) -> Tuple[dict, bool]:
    """
    Return (possibly filled spec, autofill_used).
    Conservative fill only for missing magnitude fields.
    """
    if not _autofill_enabled(scenario_autofill_cfg):
        return dict(spec), False

    defaults = ((scenario_autofill_cfg or {}).get("defaults", {}) or {}).get(control_name)
    if not isinstance(defaults, dict):
        return dict(spec), False

    out = dict(spec)
    used = False
    for k in ("from", "to", "value", "multiplier"):
        if out.get(k) is None and defaults.get(k) is not None:
            out[k] = defaults.get(k)
            used = True
    return out, used


def summarize_control_resolution(scenario_cfg: dict, scenario_autofill_cfg: Optional[dict] = None) -> dict:
    """Summarize scenario control completeness for run metadata and run notes."""
    entries = _control_entries(scenario_cfg)
    unresolved: List[dict] = []
    resolved = 0
    autofill_used_controls: List[str] = []
    for kind, name, spec in entries:
        spec_eff, used = _with_autofill_spec(name, spec, scenario_autofill_cfg)
        if used:
            autofill_used_controls.append(name)
        reason = _control_unresolved_reason(spec_eff)
        if reason is None:
            resolved += 1
        else:
            unresolved.append({"kind": kind, "control": name, "reason": reason})
    return {
        "total_controls": len(entries),
        "resolved_controls": resolved,
        "unresolved_controls": unresolved,
        "autofill_enabled": _autofill_enabled(scenario_autofill_cfg),
        "autofill_used_controls": sorted(set(autofill_used_controls)),
    }


def _control_unresolved_reason(spec: dict) -> Optional[str]:
    shape = str(spec.get("shape", "step"))
    has_from = spec.get("from") is not None
    has_to = spec.get("to") is not None
    has_value = spec.get("value") is not None
    has_multiplier = spec.get("multiplier") is not None

    if shape == "ramp":
        start = _to_float(spec.get("start_year"))
        end = _to_float(spec.get("end_year"))
        if start is None or end is None:
            return "ramp requires numeric start_year and end_year"
        if not (has_from and has_to) and not has_value and not has_multiplier:
            return "ramp requires (from,to) or value or multiplier"
        if has_from and _to_float(spec.get("from")) is None:
            return "ramp 'from' is non-numeric"
        if has_to and _to_float(spec.get("to")) is None:
            return "ramp 'to' is non-numeric"
        if has_value and _to_float(spec.get("value")) is None:
            return "ramp 'value' is non-numeric"
        if has_multiplier and _to_float(spec.get("multiplier")) is None:
            return "ramp 'multiplier' is non-numeric"
        return None

    if shape == "step":
        at = _to_float(spec.get("year"))
        if at is None:
            return "step requires numeric year"
        if not has_value and not has_multiplier:
            return "step requires value or multiplier"
        if has_value and _to_float(spec.get("value")) is None:
            return "step 'value' is non-numeric"
        if has_multiplier and _to_float(spec.get("multiplier")) is None:
            return "step 'multiplier' is non-numeric"
        return None

    if shape == "pulse":
        start = _to_float(spec.get("start_year"))
        dur = _to_float(spec.get("duration_years"))
        if start is None or dur is None or dur <= 0:
            return "pulse requires numeric start_year and positive duration_years"
        if not has_value and not has_multiplier:
            return "pulse requires value or multiplier"
        if has_value and _to_float(spec.get("value")) is None:
            return "pulse 'value' is non-numeric"
        if has_multiplier and _to_float(spec.get("multiplier")) is None:
            return "pulse 'multiplier' is non-numeric"
        return None

    return f"unsupported shape '{shape}'"


def _bound_value(control_name: str, v: float) -> float:
    if "_0_1" in control_name:
        return float(np.clip(v, 0.0, 1.0))
    if "_ge_1" in control_name:
        return float(max(v, 1.0))
    return float(v)


def _bound_multiplier(control_name: str, v: float) -> float:
    if "_ge_1" in control_name:
        return float(max(v, 1.0))
    return float(max(v, 0.0))


def _bound_output(control_name: str, v: pd.Series) -> pd.Series:
    if "_0_1" in control_name:
        return v.clip(lower=0.0, upper=1.0)
    if "_ge_1" in control_name:
        return v.clip(lower=1.0)
    return v


def _resolve_control_mode_and_value(
    control_name: str,
    spec: dict,
    year: int,
    forced_mode: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[float], str]:
    """
    Returns (is_active_this_year, mode, value, reason_if_not_applicable).
    mode in {"value", "multiplier"} when applicable.
    """
    shape = str(spec.get("shape", "step"))

    def _pick_mode() -> Optional[str]:
        if forced_mode in ("value", "multiplier"):
            return forced_mode
        if spec.get("multiplier") is not None:
            return "multiplier"
        if spec.get("from") is not None and spec.get("to") is not None:
            return "value"
        if spec.get("value") is not None:
            return "value"
        return None

    mode = _pick_mode()
    if mode is None:
        return True, None, None, "missing magnitude"

    if shape == "ramp":
        start = int(_to_float(spec.get("start_year")) or 0)
        end = int(_to_float(spec.get("end_year")) or start)
        if year < start:
            return False, None, None, "outside_window"

        if mode == "value":
            if spec.get("from") is not None and spec.get("to") is not None:
                f0 = _to_float(spec.get("from"))
                f1 = _to_float(spec.get("to"))
                if f0 is None or f1 is None:
                    return True, None, None, "non_numeric_from_to"
                frac = 1.0 if year >= end else (year - start) / max(end - start, 1)
                return True, "value", _bound_value(control_name, f0 + (f1 - f0) * frac), ""
            val = _to_float(spec.get("value"))
            if val is None:
                return True, None, None, "missing_value"
            return True, "value", _bound_value(control_name, val), ""

        # mode == multiplier
        if spec.get("from") is not None and spec.get("to") is not None:
            f0 = _to_float(spec.get("from"))
            f1 = _to_float(spec.get("to"))
            if f0 is None or f1 is None:
                return True, None, None, "non_numeric_from_to"
            frac = 1.0 if year >= end else (year - start) / max(end - start, 1)
            return True, "multiplier", _bound_multiplier(control_name, f0 + (f1 - f0) * frac), ""
        mul = _to_float(spec.get("multiplier"))
        if mul is None:
            mul = _to_float(spec.get("value"))
        if mul is None:
            return True, None, None, "missing_multiplier"
        return True, "multiplier", _bound_multiplier(control_name, mul), ""

    if shape == "step":
        at = int(_to_float(spec.get("year")) or 0)
        if year < at:
            return False, None, None, "outside_window"
        raw = _to_float(spec.get("multiplier")) if mode == "multiplier" else _to_float(spec.get("value"))
        if raw is None and mode == "multiplier":
            raw = _to_float(spec.get("value"))
        if raw is None and mode == "value":
            raw = _to_float(spec.get("multiplier"))
        if raw is None:
            return True, None, None, "missing_step_magnitude"
        if mode == "multiplier":
            return True, mode, _bound_multiplier(control_name, raw), ""
        return True, mode, _bound_value(control_name, raw), ""

    if shape == "pulse":
        start = int(_to_float(spec.get("start_year")) or 0)
        dur = int(_to_float(spec.get("duration_years")) or 0)
        if not (start <= year < start + max(dur, 0)):
            return False, None, None, "outside_window"
        raw = _to_float(spec.get("multiplier")) if mode == "multiplier" else _to_float(spec.get("value"))
        if raw is None and mode == "multiplier":
            raw = _to_float(spec.get("value"))
        if raw is None and mode == "value":
            raw = _to_float(spec.get("multiplier"))
        if raw is None:
            return True, None, None, "missing_pulse_magnitude"
        if mode == "multiplier":
            return True, mode, _bound_multiplier(control_name, raw), ""
        return True, mode, _bound_value(control_name, raw), ""

    return True, None, None, f"unsupported_shape:{shape}"


def _target_mask(df: pd.DataFrame, targets: dict) -> Tuple[pd.Series, str]:
    if df.empty:
        return pd.Series([], dtype=bool), "empty_df"
    if not targets:
        return pd.Series(True, index=df.index), ""

    key_to_cols = {
        "regions": ["r"],
        "materials": ["m"],
        "end_uses": ["j"],
        "j": ["j"],
        "jd": ["jd"],
        "stages": ["p"],
        "commodities": ["c"],
        "origin_regions": ["o"],
        "destination_regions": ["d"],
        "suppliers": ["i", "o"],
    }
    mask = pd.Series(True, index=df.index)
    unmatched_keys: List[str] = []
    for key, raw_values in targets.items():
        if raw_values is None:
            continue
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        if "*" in values:
            continue
        cols = key_to_cols.get(str(key), [str(key)])
        col = next((c for c in cols if c in df.columns), None)
        if col is None:
            unmatched_keys.append(str(key))
            continue
        mask &= df[col].isin(values)

    if unmatched_keys:
        return pd.Series(False, index=df.index), f"unmatched_targets:{','.join(unmatched_keys)}"
    return mask, ""


def apply_scenario_controls_year_df(
    df: Optional[pd.DataFrame],
    target_var: str,
    year: int,
    scenario_cfg: dict,
    scenario_autofill_cfg: Optional[dict] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> tuple[Optional[pd.DataFrame], dict]:
    """
    Apply all matching scenario controls for a single variable/year dataframe.

    `aliases` maps control names to a forced mode, e.g.
    {"demand_multiplier_0_1": "multiplier"} when target_var is demand_kt_per_yr.
    """
    if df is None:
        return None, {"applied": 0, "matched_controls": 0, "skipped_unresolved": 0}
    if df.empty or "value" not in df.columns:
        return df, {"applied": 0, "matched_controls": 0, "skipped_unresolved": 0}

    out = df.copy()
    aliases = aliases or {}
    matched = 0
    applied = 0
    skipped_unresolved = 0
    autofill_used_controls: List[str] = []

    for _kind, control_name, spec in _control_entries(scenario_cfg):
        if control_name != target_var and control_name not in aliases:
            continue
        matched += 1
        spec_eff, used = _with_autofill_spec(control_name, spec, scenario_autofill_cfg)
        if used:
            autofill_used_controls.append(control_name)

        forced_mode = aliases.get(control_name)
        active, mode, magnitude, reason = _resolve_control_mode_and_value(
            control_name=control_name,
            spec=spec_eff,
            year=year,
            forced_mode=forced_mode,
        )
        if not active:
            continue
        if mode is None or magnitude is None:
            skipped_unresolved += 1
            continue

        targets = spec_eff.get("targets", {}) or {}
        mask, mask_reason = _target_mask(out, targets)
        if mask_reason.startswith("unmatched_targets"):
            skipped_unresolved += 1
            continue
        if mask.sum() == 0:
            continue

        base = pd.to_numeric(out["value"], errors="coerce")
        if mode == "multiplier":
            out.loc[mask, "value"] = _bound_output(control_name, base.loc[mask] * magnitude)
        else:
            out.loc[mask, "value"] = magnitude
        applied += int(mask.sum())

    return out, {
        "applied": applied,
        "matched_controls": matched,
        "skipped_unresolved": skipped_unresolved,
        "autofill_used_controls": sorted(set(autofill_used_controls)),
    }
