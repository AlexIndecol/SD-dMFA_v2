from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd


_EXECUTION_MODE = "native_required"


@dataclass
class SDOutputs:
    demand_kt_per_yr: Any  # long-form dataframe: t,r,m,j,value
    capacity_stage_raw_kt_per_yr: Any  # t,r,m,p,value
    lifetime_multiplier_ge_1: Any  # t,r,m,j,value
    price_index_rel: Optional[Any] = None


def _import_bptk_module() -> Optional[Any]:
    for mod_name in ("BPTK_Py", "bptk_py"):
        try:
            return __import__(mod_name)
        except Exception:
            continue
    return None


def _require_bptk_model_cls(native_backend: Any) -> Any:
    model_cls = getattr(native_backend, "Model", None)
    if model_cls is None:
        raise RuntimeError("BPTK-Py backend unavailable for native SD execution.")
    return model_cls


def _resolve_native_sd_config(configs: Dict[str, Any]) -> tuple[str, str]:
    coupling_cfg = configs.get("coupling", {}) or {}
    sd_module_cfg = (coupling_cfg.get("modules", {}) or {}).get("sd", {}) or {}
    engine_requested = str(sd_module_cfg.get("engine", "bptk-py")).strip().lower()
    mode_requested = str(sd_module_cfg.get("execution_mode", _EXECUTION_MODE)).strip().lower()

    if engine_requested != "bptk-py":
        raise ValueError(
            f"Unsupported SD engine '{engine_requested}'. "
            "Full native runtime requires coupling.modules.sd.engine='bptk-py'."
        )
    if mode_requested != _EXECUTION_MODE:
        raise ValueError(
            f"Unsupported SD execution mode '{mode_requested}'. "
            "Fallback execution mode has been removed; use execution_mode='native_required'."
        )
    return engine_requested, mode_requested


def build_sd_model(configs: Dict[str, Any]) -> Any:
    """Build strict native SD runtime contract."""
    engine_requested, execution_mode = _resolve_native_sd_config(configs)
    bptk_mod = _import_bptk_module()
    if bptk_mod is None:
        raise RuntimeError(
            "SD native engine required but BPTK-Py is unavailable. "
            "Install module 'BPTK_Py' in the active environment."
        )
    return {
        "engine": engine_requested,
        "execution_mode": execution_mode,
        "native_available": True,
        "use_native": True,
        "native_backend": bptk_mod,
        "configs": configs,
    }


def _year(df: Optional[pd.DataFrame], year: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "t" in df.columns:
        return df[df["t"] == year].copy()
    return df.copy()


def _build_price(base_rm: pd.DataFrame, year: int, price_in: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = base_rm.copy()
    out["t"] = year
    out["value"] = 1.0
    p = _year(price_in, year)
    if p.empty:
        return out[["t", "r", "m", "value"]]
    keys = [c for c in ["r", "m"] if c in p.columns]
    p2 = p.groupby(keys, as_index=False)["value"].mean()
    out = out.merge(p2, on=keys, how="left", suffixes=("", "_p"))
    out["value"] = out["value_p"].fillna(out["value"])
    return out[["t", "r", "m", "value"]]


def _baseline_global_demand(materials: list[str], year: int, sd_cfg: Dict[str, Any]) -> dict[str, float]:
    dcfg = sd_cfg.get("demand", {})
    base = dcfg.get("baseline_1970_global_demand_kt_per_yr", {}) or {}
    g1 = dcfg.get("baseline_growth_cagr_1970_2000", {}) or {}
    g2 = dcfg.get("baseline_growth_cagr_2000_2050", {}) or {}

    out: dict[str, float] = {}
    for m in materials:
        d0 = float(base.get(m, 0.0))
        if year <= 1970:
            out[m] = d0
            continue
        y1 = min(max(year, 1970), 2000) - 1970
        y2 = max(year - 2000, 0)
        out[m] = d0 * ((1.0 + float(g1.get(m, 0.0))) ** y1) * ((1.0 + float(g2.get(m, 0.0))) ** y2)
    return out


def _default_use_share(base_rmj: pd.DataFrame, sd_cfg: Dict[str, Any], year: int) -> pd.DataFrame:
    rows = []
    share_cfg = (
        sd_cfg.get("demand", {})
        .get("end_use_share_default_by_material", {})
        or {}
    )
    for (r, m), g in base_rmj.groupby(["r", "m"]):
        shares = share_cfg.get(m, {}) or {}
        vals = []
        for j in g["j"].tolist():
            vals.append(float(shares.get(j, 0.0)))
        s = sum(vals)
        if s <= 0:
            vals = [1.0 / len(g)] * len(g)
        else:
            vals = [v / s for v in vals]
        for j, v in zip(g["j"], vals):
            rows.append({"t": year, "r": r, "m": m, "j": j, "value": v})
    return pd.DataFrame(rows)


def _make_bptk_price_adjust(native_backend: Any) -> Callable[[float, float, float], float]:
    model_cls = _require_bptk_model_cls(native_backend)

    def _adjust(demand_value: float, price_value: float, elasticity: float) -> float:
        mdl = model_cls(starttime=0.0, stoptime=0.0, dt=1.0, name="sd_price_adjust")
        d = mdl.constant("d")
        p = mdl.constant("p")
        e = mdl.constant("e")
        y = mdl.converter("y")
        d.equation = float(demand_value)
        p.equation = float(max(float(price_value), 1e-6))
        e.equation = float(elasticity)
        y.equation = d * (p**e)
        return float(mdl.equation("y", 0.0))

    return _adjust


def _make_bptk_scale(native_backend: Any) -> Callable[[float, float], float]:
    model_cls = _require_bptk_model_cls(native_backend)

    def _scale(base_value: float, scale_value: float) -> float:
        mdl = model_cls(starttime=0.0, stoptime=0.0, dt=1.0, name="sd_scale")
        b = mdl.constant("b")
        s = mdl.constant("s")
        y = mdl.converter("y")
        b.equation = float(base_value)
        s.equation = float(scale_value)
        y.equation = b * s
        return float(mdl.equation("y", 0.0))

    return _scale


def _make_bptk_identity(native_backend: Any) -> Callable[[float], float]:
    model_cls = _require_bptk_model_cls(native_backend)

    def _identity(value: float) -> float:
        mdl = model_cls(starttime=0.0, stoptime=0.0, dt=1.0, name="sd_identity")
        x = mdl.constant("x")
        y = mdl.converter("y")
        x.equation = float(value)
        y.equation = x
        return float(mdl.equation("y", 0.0))

    return _identity


def _run_sd_step_impl(
    model: Any,
    inputs: Dict[str, Any],
    year: int,
    *,
    price_adjust: Callable[[float, float, float], float],
    scale_fn: Callable[[float, float], float],
    identity_fn: Callable[[float], float],
) -> SDOutputs:
    cfg = model.get("configs", {})
    sd_cfg = cfg.get("parameters", {}).get("sd", {}) or {}
    dims = cfg.get("dimensions", {}).get("dimensions", {}) or {}

    base_rmj = inputs.get("base_rmj")
    base_rmp = inputs.get("base_rmp")
    if base_rmj is None or base_rmp is None or base_rmj.empty or base_rmp.empty:
        raise ValueError("SD step requires non-empty base grids: base_rmj and base_rmp")

    # Demand from observed GAS if available, otherwise from baseline growth + shares.
    gas = _year(inputs.get("gas_to_use_observed_kt_per_yr"), year)
    demand = base_rmj.copy()
    demand["t"] = year
    demand["value"] = 0.0

    gas_has_numeric = (
        not gas.empty
        and {"r", "m", "j", "value"}.issubset(gas.columns)
        and pd.to_numeric(gas["value"], errors="coerce").notna().any()
    )

    if gas_has_numeric:
        gas2 = gas.copy()
        gas2["value"] = pd.to_numeric(gas2["value"], errors="coerce")
        gas2 = gas2.groupby(["r", "m", "j"], as_index=False)["value"].sum(min_count=1)
        demand = demand.merge(gas2, on=["r", "m", "j"], how="left", suffixes=("", "_obs"))
        demand["value"] = demand["value_obs"].fillna(0.0)
        demand = demand[["t", "r", "m", "j", "value"]]
    else:
        prev = inputs.get("last_i_use")
        if prev is not None and not prev.empty:
            p = prev.groupby(["r", "m", "j"], as_index=False)["value"].sum()
            demand = demand.merge(p, on=["r", "m", "j"], how="left", suffixes=("", "_prev"))
            demand["value"] = demand["value_prev"].fillna(0.0)
            demand = demand[["t", "r", "m", "j", "value"]]
        else:
            mats = dims.get("materials", {}).get("values", []) or sorted(base_rmj["m"].unique().tolist())
            baseline = _baseline_global_demand(mats, year, sd_cfg)
            use_share = _year(inputs.get("use_share_j_frac"), year)
            if use_share.empty:
                use_share = _default_use_share(base_rmj, sd_cfg, year)
            u = use_share.groupby(["m", "j"], as_index=False)["value"].mean()
            demand = demand.merge(u, on=["m", "j"], how="left", suffixes=("", "_share"))
            demand["value_share"] = demand["value_share"].fillna(0.0)
            demand["regions_n"] = demand.groupby(["m", "j"])["r"].transform("nunique").clip(lower=1)
            demand["baseline_m"] = demand["m"].map(baseline).fillna(0.0)
            demand["value"] = demand["baseline_m"] * demand["value_share"] / demand["regions_n"]
            demand = demand[["t", "r", "m", "j", "value"]]

    # Price feedback.
    elasticity = (
        sd_cfg.get("price_module", {})
        .get("demand_price_elasticity_short_run", {})
        or {}
    )
    if not demand.empty:
        rm = demand[["r", "m"]].drop_duplicates()
        price = _build_price(rm, year, inputs.get("price_to_sd_smoothed_rel"))
        demand = demand.merge(price, on=["t", "r", "m"], how="left", suffixes=("", "_price"))
        demand["value_price"] = demand["value_price"].fillna(1.0)
        demand["elast"] = pd.to_numeric(demand["m"].map(elasticity), errors="coerce").fillna(0.0)
        demand["value"] = demand.apply(
            lambda x: price_adjust(float(x["value"]), float(x["value_price"]), float(x["elast"])),
            axis=1,
        )
        demand = demand[["t", "r", "m", "j", "value"]]
        demand["value"] = demand["value"].map(identity_fn)

    # Capacity from observed exogenous if provided, else infer from demand total.
    cap_obs = _year(inputs.get("capacity_stage_observed_kt_per_yr"), year)
    cap = base_rmp.copy()
    cap["t"] = year
    cap["value"] = 0.0
    if not cap_obs.empty and {"r", "m", "p", "value"}.issubset(cap_obs.columns):
        c = cap_obs.groupby(["r", "m", "p"], as_index=False)["value"].sum()
        cap = cap.merge(c, on=["r", "m", "p"], how="left", suffixes=("", "_obs"))
        cap["value"] = cap["value_obs"].fillna(0.0).map(identity_fn)
        cap = cap[["t", "r", "m", "p", "value"]]
    else:
        dem_rm = demand.groupby(["r", "m"], as_index=False)["value"].sum().rename(columns={"value": "dem"})
        cap = cap.merge(dem_rm, on=["r", "m"], how="left")
        slack = float(sd_cfg.get("capacity", {}).get("initial_slack_factor", 1.1))
        cap["value"] = cap["dem"].fillna(0.0).map(lambda v: scale_fn(float(v), slack))
        cap = cap[["t", "r", "m", "p", "value"]]

    # R-strategy rates by region/material (passed as side-channel to DMFA).
    rr_base = (
        sd_cfg.get("r_strategies", {})
        .get("baseline_targets_2020", {})
        .get("eol_recycling_rate_frac", {})
        or {}
    )
    coll_base = (
        sd_cfg.get("r_strategies", {})
        .get("baseline_targets_2020", {})
        .get("collection_rate_frac", {})
        or {}
    )
    base_rm = base_rmj[["r", "m"]].drop_duplicates()

    collection = base_rm.copy()
    collection["t"] = year
    collection["value"] = collection["r"].map(lambda r: float(coll_base.get(r, 0.5))).clip(0.0, 1.0).map(identity_fn)

    recovery = base_rm.copy()
    recovery["t"] = year
    recovery["value"] = recovery.apply(
        lambda x: float((rr_base.get(x["r"], {}) or {}).get(x["m"], 0.4)), axis=1
    ).clip(0.0, 1.0).map(identity_fn)

    recycling_yield = base_rm.copy()
    recycling_yield["t"] = year
    recycling_yield["value"] = 0.9
    recycling_yield["value"] = recycling_yield["value"].map(identity_fn)

    life_mult = base_rmj.copy()
    life_mult["t"] = year
    life_mult["value"] = 1.0
    life_mult["value"] = life_mult["value"].map(identity_fn)

    price_out = _build_price(base_rm, year, inputs.get("price_to_sd_smoothed_rel"))
    price_out["value"] = price_out["value"].map(identity_fn)

    inputs["collection_rate_0_1_year"] = collection[["t", "r", "m", "value"]]
    inputs["recovery_rate_0_1_year"] = recovery[["t", "r", "m", "value"]]
    inputs["recycling_yield_0_1_year"] = recycling_yield[["t", "r", "m", "value"]]

    return SDOutputs(
        demand_kt_per_yr=demand[["t", "r", "m", "j", "value"]],
        capacity_stage_raw_kt_per_yr=cap[["t", "r", "m", "p", "value"]],
        lifetime_multiplier_ge_1=life_mult[["t", "r", "m", "j", "value"]],
        price_index_rel=price_out[["t", "r", "m", "value"]],
    )


def _run_sd_step_native_bptk(model: Any, inputs: Dict[str, Any], year: int) -> SDOutputs:
    adjust = _make_bptk_price_adjust(model.get("native_backend"))
    scale_fn = _make_bptk_scale(model.get("native_backend"))
    identity_fn = _make_bptk_identity(model.get("native_backend"))
    return _run_sd_step_impl(
        model=model,
        inputs=inputs,
        year=year,
        price_adjust=adjust,
        scale_fn=scale_fn,
        identity_fn=identity_fn,
    )


def run_sd_step(model: Any, inputs: Dict[str, Any], year: int) -> SDOutputs:
    """Run one SD timestep with native BPTK execution."""
    if bool(model.get("use_native", False)) and model.get("engine") == "bptk-py":
        return _run_sd_step_native_bptk(model, inputs, year)
    raise RuntimeError(
        "SD native runtime unavailable. Full native migration requires BPTK-Py. "
        "Ensure coupling.modules.sd.execution_mode='native_required' and dependency installed."
    )
