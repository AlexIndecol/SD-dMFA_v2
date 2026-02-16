from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .sd_model import build_sd_model, run_sd_step
from .dmfa_model import build_dmfa_model, run_dmfa_step
from .trade_allocator import allocate_od_weights
from .scenario_controls import apply_scenario_controls_year_df


def smooth_exponential(prev: Optional[pd.DataFrame], raw: pd.DataFrame, tau_years: float, keys: list[str]) -> pd.DataFrame:
    """First-order exponential smoothing for long-form series."""
    if prev is None or prev.empty:
        return raw.copy()
    p = prev.rename(columns={"value": "prev"})
    r = raw.rename(columns={"value": "raw"})
    m = r.merge(p, on=keys, how="left")
    m["prev"] = m["prev"].fillna(m["raw"])
    m["value"] = m["prev"] + (m["raw"] - m["prev"]) / max(tau_years, 1e-6)
    return m[keys + ["value"]]


def _year(df: Optional[pd.DataFrame], year: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "t" in df.columns:
        return df[df["t"] == year].copy()
    return df.copy()


def _drop_t(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "t" in out.columns:
        out = out.drop(columns=["t"])
    return out


def _append_t(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out["t"] = year
    cols = ["t"] + [c for c in out.columns if c != "t"]
    return out[cols]


def _compute_share_i(flows: pd.DataFrame) -> pd.DataFrame:
    if flows.empty:
        return pd.DataFrame(columns=["t", "r", "m", "i", "value"])
    x = flows.groupby(["t", "d", "m", "o"], as_index=False)["value"].sum().rename(columns={"d": "r", "o": "i"})
    tot = x.groupby(["t", "r", "m"], as_index=False)["value"].sum().rename(columns={"value": "tot"})
    x = x.merge(tot, on=["t", "r", "m"], how="left")
    x["value"] = np.where(x["tot"] > 0, x["value"] / x["tot"], 0.0)
    return x[["t", "r", "m", "i", "value"]]


@dataclass
class CoupledState:
    sd_model: Any
    dmfa_model: Any
    ceilings: Dict[str, pd.DataFrame]
    memory: Dict[str, pd.DataFrame]


def build_coupled_system(configs: Dict[str, Any]) -> CoupledState:
    return CoupledState(
        sd_model=build_sd_model(configs),
        dmfa_model=build_dmfa_model(configs),
        ceilings={},
        memory={},
    )


def run_coupled_year(state: CoupledState, exogenous: Dict[str, pd.DataFrame], year: int, configs: Dict[str, Any]) -> Dict[str, Any]:
    """One annual coupled step: SD -> smoothing -> dMFA -> OD trade."""
    tcfg = configs.get("time", {}).get("stabilizers", {}) or {}
    tau = float(tcfg.get("capacity_ceiling_smoothing_tau_years", 2.0))
    price_tau = float(tcfg.get("price_smoothing_tau_years", 1.0))
    scenario_cfg = configs.get("scenario", {}) or {}
    scenario_autofill_cfg = configs.get("scenario_autofill", {}) or {}

    base_rmj = exogenous.get("_base_rmj")
    base_rmp = exogenous.get("_base_rmp")
    base_rmc = exogenous.get("_base_rmc")
    if base_rmj is None or base_rmp is None or base_rmc is None:
        raise ValueError("Missing base grids in exogenous bundle (_base_rmj, _base_rmp, _base_rmc)")

    # 1) SD step
    gas_year, _ = apply_scenario_controls_year_df(
        _year(exogenous.get("gas_to_use_observed_kt_per_yr"), year),
        target_var="gas_to_use_observed_kt_per_yr",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    use_share_year, _ = apply_scenario_controls_year_df(
        _year(exogenous.get("use_share_j_frac"), year),
        target_var="use_share_j_frac",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    cap_obs_year, _ = apply_scenario_controls_year_df(
        _year(exogenous.get("capacity_stage_observed_kt_per_yr"), year),
        target_var="capacity_stage_observed_kt_per_yr",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    sd_inputs: Dict[str, Any] = {
        "base_rmj": base_rmj,
        "base_rmp": base_rmp,
        "gas_to_use_observed_kt_per_yr": gas_year,
        "use_share_j_frac": use_share_year,
        "capacity_stage_observed_kt_per_yr": cap_obs_year,
        "price_to_sd_smoothed_rel": state.ceilings.get("price_to_sd_smoothed_rel"),
        "last_i_use": state.memory.get("last_i_use_kt_per_yr"),
    }
    sd_out = run_sd_step(state.sd_model, sd_inputs, year)

    # Apply scenario controls that target runtime SD outputs / SD->dMFA exchanges.
    sd_out.demand_kt_per_yr, _ = apply_scenario_controls_year_df(
        sd_out.demand_kt_per_yr,
        target_var="demand_kt_per_yr",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
        aliases={
            "demand_multiplier_0_1": "multiplier",
            "demand_multiplier_ge_1": "multiplier",
        },
    )
    sd_inputs["collection_rate_0_1_year"], _ = apply_scenario_controls_year_df(
        sd_inputs.get("collection_rate_0_1_year"),
        target_var="collection_rate_0_1",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    sd_inputs["recovery_rate_0_1_year"], _ = apply_scenario_controls_year_df(
        sd_inputs.get("recovery_rate_0_1_year"),
        target_var="recovery_rate_0_1",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    sd_inputs["recycling_yield_0_1_year"], _ = apply_scenario_controls_year_df(
        sd_inputs.get("recycling_yield_0_1_year"),
        target_var="recycling_yield_0_1",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    sd_out.lifetime_multiplier_ge_1, _ = apply_scenario_controls_year_df(
        sd_out.lifetime_multiplier_ge_1,
        target_var="lifetime_multiplier_ge_1",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    sd_out.capacity_stage_raw_kt_per_yr, _ = apply_scenario_controls_year_df(
        sd_out.capacity_stage_raw_kt_per_yr,
        target_var="capacity_stage_raw_kt_per_yr",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    sd_out.price_index_rel, _ = apply_scenario_controls_year_df(
        sd_out.price_index_rel,
        target_var="price_index_rel",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )

    # 2) Stabilize shared ceilings
    cap_raw = _drop_t(sd_out.capacity_stage_raw_kt_per_yr)
    cap_sm = smooth_exponential(
        state.ceilings.get("capacity_stage_to_sd_kt_per_yr"),
        cap_raw,
        tau_years=tau,
        keys=["r", "m", "p"],
    )
    cap_sm = _append_t(cap_sm, year)
    state.ceilings["capacity_stage_to_sd_kt_per_yr"] = _drop_t(cap_sm)

    price_raw = _drop_t(sd_out.price_index_rel) if sd_out.price_index_rel is not None else pd.DataFrame(columns=["r", "m", "value"])
    price_sm = smooth_exponential(
        state.ceilings.get("price_to_sd_smoothed_rel"),
        price_raw,
        tau_years=price_tau,
        keys=["r", "m"],
    ) if not price_raw.empty else pd.DataFrame(columns=["r", "m", "value"])
    if not price_sm.empty:
        price_sm = _append_t(price_sm, year)
        state.ceilings["price_to_sd_smoothed_rel"] = _drop_t(price_sm)

    # 3) dMFA step
    dmfa_inputs: Dict[str, Any] = {
        "base_rmj": base_rmj,
        "base_rmc": base_rmc,
        "demand_kt_per_yr_year": sd_out.demand_kt_per_yr,
        "capacity_stage_to_sd_kt_per_yr_year": cap_sm,
        "lifetime_multiplier_ge_1_year": sd_out.lifetime_multiplier_ge_1,
        "collection_rate_0_1_year": sd_inputs.get("collection_rate_0_1_year"),
        "recovery_rate_0_1_year": sd_inputs.get("recovery_rate_0_1_year"),
        "recycling_yield_0_1_year": sd_inputs.get("recycling_yield_0_1_year"),
        "lifetime_mean_years": exogenous.get("lifetime_mean_years"),
        "lifetime_lognormal_mu": exogenous.get("lifetime_lognormal_mu"),
        "lifetime_lognormal_sigma": exogenous.get("lifetime_lognormal_sigma"),
        "ore_mined_kt_per_yr": exogenous.get("ore_mined_kt_per_yr"),
        "last_in_use_stock_kt": state.memory.get("last_in_use_stock_kt"),
        "last_stock_kt": state.memory.get("last_stock_kt"),
        "last_imports_kt_per_yr": state.memory.get("last_imports_kt_per_yr"),
        "last_exports_kt_per_yr": state.memory.get("last_exports_kt_per_yr"),
    }
    dmfa_out = run_dmfa_step(state.dmfa_model, dmfa_inputs, year)

    # 3b) Smooth secondary-availability ceiling and export cap.
    ssec_raw = _drop_t(dmfa_out.secondary_production_kt_per_yr)
    ssec_sm = smooth_exponential(
        state.ceilings.get("s_sec_max_to_sd_kt_per_yr"),
        ssec_raw,
        tau_years=tau,
        keys=["r", "m"],
    )
    ssec_sm = _append_t(ssec_sm, year)
    state.ceilings["s_sec_max_to_sd_kt_per_yr"] = _drop_t(ssec_sm)

    exp_raw = _drop_t(dmfa_out.export_cap_kt_per_yr)
    exp_sm = smooth_exponential(
        state.ceilings.get("od_export_cap_kt_per_yr"),
        exp_raw,
        tau_years=tau,
        keys=["r", "m", "c"],
    )
    exp_sm = _append_t(exp_sm, year)
    state.ceilings["od_export_cap_kt_per_yr"] = _drop_t(exp_sm)

    # 4) OD trade allocation
    od_pref = _year(exogenous.get("od_preference_weight_0_1"), year)
    od_pref, _ = apply_scenario_controls_year_df(
        od_pref,
        target_var="od_preference_weight_0_1",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    if od_pref.empty:
        od_pref = exogenous.get("_base_od", pd.DataFrame(columns=["t", "o", "d", "m", "c", "value"])).copy()
        od_pref["t"] = year
        od_pref["value"] = 0.0

    tf_src = _year(exogenous.get("trade_factor_i_ge_1"), year)
    tf_src, _ = apply_scenario_controls_year_df(
        tf_src,
        target_var="trade_factor_i_ge_1",
        year=year,
        scenario_cfg=scenario_cfg,
        scenario_autofill_cfg=scenario_autofill_cfg,
    )
    tf_od = None
    if not tf_src.empty:
        tf = tf_src.copy()
        if "i" in tf.columns:
            tf = tf.rename(columns={"i": "o"})
        tf = tf[["t", "o", "m", "c", "value"]]
        tf_od = od_pref[["t", "o", "d", "m", "c"]].merge(tf, on=["t", "o", "m", "c"], how="left")
        tf_od["value"] = tf_od["value"].fillna(1.0)

    desired_rm = sd_out.demand_kt_per_yr.groupby(["t", "r", "m"], as_index=False)["value"].sum().rename(columns={"value": "desired"})
    realized_rm = dmfa_out.i_use_kt_per_yr.groupby(["t", "r", "m"], as_index=False)["value"].sum().rename(columns={"value": "realized"})
    unmet = desired_rm.merge(realized_rm, on=["t", "r", "m"], how="left")
    unmet["realized"] = unmet["realized"].fillna(0.0)
    unmet["unmet"] = (unmet["desired"] - unmet["realized"]).clip(lower=0.0)

    dest_cap = base_rmc.copy()
    dest_cap["t"] = year
    dest_cap = dest_cap.merge(unmet[["r", "m", "unmet"]], on=["r", "m"], how="left")
    dest_cap["unmet"] = dest_cap["unmet"].fillna(0.0)
    dest_cap["value"] = np.where(dest_cap["c"] == "refined_metal", dest_cap["unmet"], 1e18)
    dest_cap = dest_cap[["t", "r", "m", "c", "value"]]

    trade_out = allocate_od_weights(
        year=year,
        od_pref=od_pref,
        export_cap=exp_sm,
        trade_factor=tf_od,
        destination_cap=dest_cap,
    )

    share_i = _compute_share_i(trade_out.od_trade_flow_kt_per_yr)

    # 5) Persist state for next year
    prev_in_use = state.memory.get("last_in_use_stock_kt")
    if prev_in_use is None or prev_in_use.empty:
        prev_in_use = base_rmj.copy()
        prev_in_use["value"] = 0.0

    i_use_no_t = _drop_t(dmfa_out.i_use_kt_per_yr).rename(columns={"value": "i_use"})
    eol_no_t = _drop_t(dmfa_out.eol_outflow_kt_per_yr).rename(columns={"value": "eol"})

    in_use_new = prev_in_use.merge(i_use_no_t, on=["r", "m", "j"], how="left")
    in_use_new = in_use_new.merge(eol_no_t, on=["r", "m", "j"], how="left")
    in_use_new["i_use"] = in_use_new["i_use"].fillna(0.0)
    in_use_new["eol"] = in_use_new["eol"].fillna(0.0)
    in_use_new["value"] = (in_use_new["value"] + in_use_new["i_use"] - in_use_new["eol"]).clip(lower=0.0)
    in_use_new = in_use_new[["r", "m", "j", "value"]]

    state.memory["last_i_use_kt_per_yr"] = _drop_t(dmfa_out.i_use_kt_per_yr)
    state.memory["last_in_use_stock_kt"] = in_use_new
    state.memory["last_stock_kt"] = _drop_t(dmfa_out.stock_kt)
    state.memory["last_imports_kt_per_yr"] = _drop_t(trade_out.imports_kt_per_yr)
    state.memory["last_exports_kt_per_yr"] = _drop_t(trade_out.exports_kt_per_yr)

    # Commodity stock convenience series.
    stock_ref = dmfa_out.stock_kt[dmfa_out.stock_kt["c"] == "refined_metal"][["t", "r", "m", "value"]].copy()
    stock_scr = dmfa_out.stock_kt[dmfa_out.stock_kt["c"] == "scrap"][["t", "r", "m", "value"]].copy()
    stock_con = dmfa_out.stock_kt[dmfa_out.stock_kt["c"] == "concentrate"][["t", "r", "m", "value"]].copy()

    # Domestic production proxy for apparent-consumption accounting:
    # map primary+secondary refined production to commodity c='refined_metal'.
    dom_prod = dmfa_out.primary_production_kt_per_yr.merge(
        dmfa_out.secondary_production_kt_per_yr,
        on=["t", "r", "m"],
        how="outer",
        suffixes=("_prim", "_sec"),
    )
    dom_prod["value_prim"] = pd.to_numeric(dom_prod["value_prim"], errors="coerce").fillna(0.0)
    dom_prod["value_sec"] = pd.to_numeric(dom_prod["value_sec"], errors="coerce").fillna(0.0)
    dom_prod["value"] = dom_prod["value_prim"] + dom_prod["value_sec"]
    dom_prod["c"] = "refined_metal"
    dom_prod = dom_prod[["t", "r", "m", "c", "value"]]

    # Realized unmet demand.
    unmet_series = unmet[["t", "r", "m", "unmet"]].rename(columns={"unmet": "value"})

    outputs: Dict[str, Any] = {
        "demand_kt_per_yr": sd_out.demand_kt_per_yr,
        "capacity_stage_raw_kt_per_yr": sd_out.capacity_stage_raw_kt_per_yr,
        "capacity_stage_to_sd_kt_per_yr": cap_sm,
        "price_index_rel": sd_out.price_index_rel,
        "price_to_sd_smoothed_rel": price_sm if not price_sm.empty else pd.DataFrame(columns=["t", "r", "m", "value"]),
        "lifetime_multiplier_ge_1": sd_out.lifetime_multiplier_ge_1,
        "collection_rate_0_1": sd_inputs.get("collection_rate_0_1_year"),
        "recovery_rate_0_1": sd_inputs.get("recovery_rate_0_1_year"),
        "recycling_yield_0_1": sd_inputs.get("recycling_yield_0_1_year"),
        "i_use_kt_per_yr": dmfa_out.i_use_kt_per_yr,
        "in_use_stock_kt": _append_t(in_use_new, year),
        "eol_outflow_kt_per_yr": dmfa_out.eol_outflow_kt_per_yr,
        "primary_production_kt_per_yr": dmfa_out.primary_production_kt_per_yr,
        "secondary_production_kt_per_yr": dmfa_out.secondary_production_kt_per_yr,
        "eol_recycling_kt_per_yr": dmfa_out.eol_recycling_kt_per_yr,
        "apparent_consumption_kt_per_yr": dmfa_out.apparent_consumption_kt_per_yr,
        "domestic_production_kt_per_yr": dom_prod,
        "stock_change_kt_per_yr": dmfa_out.stock_change_kt_per_yr,
        "balancing_item_kt_per_yr": dmfa_out.balancing_item_kt_per_yr,
        "negativity_clipped_kt_per_yr": dmfa_out.negativity_clipped_kt_per_yr,
        "new_scrap_generated_kt_per_yr": dmfa_out.new_scrap_generated_kt_per_yr,
        "old_scrap_generated_kt_per_yr": dmfa_out.old_scrap_generated_kt_per_yr,
        "old_scrap_recycled_kt_per_yr": dmfa_out.old_scrap_recycled_kt_per_yr,
        "primary_input_kt_per_yr": dmfa_out.primary_input_kt_per_yr,
        "secondary_input_old_scrap_kt_per_yr": dmfa_out.secondary_input_old_scrap_kt_per_yr,
        "secondary_input_total_kt_per_yr": dmfa_out.secondary_input_total_kt_per_yr,
        "process_flow_kt_per_yr": dmfa_out.process_flow_kt_per_yr,
        "scrap_buffer_kt": dmfa_out.scrap_buffer_kt,
        "scrap_release_kt_per_yr": dmfa_out.scrap_release_kt_per_yr,
        "s_sec_max_raw_kt_per_yr": dmfa_out.secondary_production_kt_per_yr,
        "s_sec_max_to_sd_kt_per_yr": ssec_sm,
        "stock_refined_metal_kt": stock_ref,
        "stock_scrap_kt": stock_scr,
        "stock_concentrate_kt": stock_con,
        "od_export_cap_kt_per_yr": exp_sm,
        "od_trade_flow_kt_per_yr": trade_out.od_trade_flow_kt_per_yr,
        "imports_kt_per_yr": trade_out.imports_kt_per_yr,
        "exports_kt_per_yr": trade_out.exports_kt_per_yr,
        "share_i_frac": share_i,
        "unmet_demand_kt_per_yr": unmet_series,
    }
    return outputs
