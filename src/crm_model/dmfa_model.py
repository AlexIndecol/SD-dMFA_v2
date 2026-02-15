from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class DMFAOutputs:
    # Core delivered demand / use inflows
    i_use_kt_per_yr: Any  # t,r,m,j,value
    eol_outflow_kt_per_yr: Any  # t,r,m,j,value

    # Commodity stocks and flows (commodity dimension c)
    stock_kt: Any  # t,r,m,c,value
    stock_change_kt_per_yr: Any  # t,r,m,c,value
    apparent_consumption_kt_per_yr: Any  # t,r,m,c,value
    balancing_item_kt_per_yr: Any  # t,r,m,c,value
    negativity_clipped_kt_per_yr: Any  # t,r,m,c,value

    # Scrap split and optional buffer
    new_scrap_generated_kt_per_yr: Any  # t,r,m,j,value
    old_scrap_generated_kt_per_yr: Any  # t,r,m,j,value
    scrap_buffer_kt: Optional[Any] = None  # t,r,m,value
    scrap_release_kt_per_yr: Optional[Any] = None  # t,r,m,value

    # Supply-side stage outputs for trade and SD feedback
    export_cap_kt_per_yr: Optional[Any] = None  # t,r,m,c,value
    primary_production_kt_per_yr: Optional[Any] = None  # t,r,m,value
    secondary_production_kt_per_yr: Optional[Any] = None  # t,r,m,value
    eol_recycling_kt_per_yr: Optional[Any] = None  # t,r,m,value


def build_dmfa_model(configs: Dict[str, Any]) -> Any:
    """Build dMFA model using flodym.

    The coupled fallback remains runnable even if flodym is unavailable.
    """
    try:
        import flodym  # type: ignore
    except Exception:
        flodym = None
    return {"engine": "flodym", "available": flodym is not None, "configs": configs}


def apply_lifetime_extension(mu: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    """Lognormal lifetime extension rule aligned with MISO2: mu' = mu + ln(k)."""
    return mu + np.log(np.maximum(multiplier, 1.0))


def _year(df: Optional[pd.DataFrame], year: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "t" in df.columns:
        return df[df["t"] == year].copy()
    return df.copy()


def _to_rm(series: Optional[pd.DataFrame], name: str, base_rm: pd.DataFrame, default: float = 0.0) -> pd.DataFrame:
    out = base_rm.copy()
    out[name] = default
    if series is None or series.empty:
        return out
    keys = [c for c in ["r", "m"] if c in series.columns]
    if not keys:
        return out
    s = series.groupby(keys, as_index=False)["value"].sum().rename(columns={"value": name})
    out = out.merge(s, on=keys, how="left", suffixes=("", "_obs"))
    out[name] = out[f"{name}_obs"].fillna(out[name])
    return out[["r", "m", name]]


def _lifetime_mean(base_rmj: pd.DataFrame, inputs: Dict[str, Any], year: int) -> pd.DataFrame:
    out = base_rmj.copy()
    out["value"] = np.nan

    mean_df = _year(inputs.get("lifetime_mean_years"), year)
    if mean_df.empty:
        mean_df = inputs.get("lifetime_mean_years") if inputs.get("lifetime_mean_years") is not None else pd.DataFrame()

    if isinstance(mean_df, pd.DataFrame) and not mean_df.empty and {"r", "m", "j", "value"}.issubset(mean_df.columns):
        m = mean_df.groupby(["r", "m", "j"], as_index=False)["value"].mean()
        out = out.merge(m, on=["r", "m", "j"], how="left", suffixes=("", "_mean"))
        out["value"] = out["value_mean"]
        out = out[["r", "m", "j", "value"]]

    if out["value"].isna().any():
        mu = inputs.get("lifetime_lognormal_mu")
        sig = inputs.get("lifetime_lognormal_sigma")
        if isinstance(mu, pd.DataFrame) and isinstance(sig, pd.DataFrame) and not mu.empty and not sig.empty:
            m2 = mu.groupby(["r", "m", "j"], as_index=False)["value"].mean().rename(columns={"value": "mu"})
            s2 = sig.groupby(["r", "m", "j"], as_index=False)["value"].mean().rename(columns={"value": "sigma"})
            z = m2.merge(s2, on=["r", "m", "j"], how="inner")
            z["mean"] = np.exp(z["mu"] + 0.5 * np.square(z["sigma"]))
            out = out.merge(z[["r", "m", "j", "mean"]], on=["r", "m", "j"], how="left")
            out["value"] = out["value"].fillna(out["mean"])
            out = out[["r", "m", "j", "value"]]

    out["value"] = out["value"].fillna(20.0).clip(lower=1.0)
    return out


def run_dmfa_step(model: Any, inputs: Dict[str, Any], year: int) -> DMFAOutputs:
    """Run one dMFA timestep with mass-balance preserving fallback equations."""
    cfg = model.get("configs", {})
    sd_cfg = cfg.get("parameters", {}).get("sd", {}) or {}

    base_rmj = inputs.get("base_rmj")
    base_rmc = inputs.get("base_rmc")
    if base_rmj is None or base_rmc is None or base_rmj.empty or base_rmc.empty:
        raise ValueError("DMFA step requires non-empty base grids: base_rmj and base_rmc")

    base_rm = base_rmj[["r", "m"]].drop_duplicates()

    demand = inputs.get("demand_kt_per_yr_year")
    if demand is None or demand.empty:
        demand = base_rmj.copy()
        demand["t"] = year
        demand["value"] = 0.0
    demand = demand[["t", "r", "m", "j", "value"]].copy()

    in_use_prev = inputs.get("last_in_use_stock_kt")
    if in_use_prev is None or in_use_prev.empty:
        in_use_prev = base_rmj.copy()
        in_use_prev["value"] = 0.0

    # Lifetime and EoL outflow.
    life = _lifetime_mean(base_rmj, inputs, year).rename(columns={"value": "life_mean"})
    life_mult = inputs.get("lifetime_multiplier_ge_1_year")
    if life_mult is None or life_mult.empty:
        life_mult = base_rmj.copy()
        life_mult["value"] = 1.0
    life_mult = life_mult[["r", "m", "j", "value"]].rename(columns={"value": "life_mult"})

    eol = in_use_prev.merge(life, on=["r", "m", "j"], how="left")
    eol = eol.merge(life_mult, on=["r", "m", "j"], how="left")
    eol["life_mean"] = eol["life_mean"].fillna(20.0).clip(lower=1.0)
    eol["life_mult"] = eol["life_mult"].fillna(1.0).clip(lower=1.0)
    eol["value"] = eol["value"] / (eol["life_mean"] * eol["life_mult"])
    eol["t"] = year
    eol = eol[["t", "r", "m", "j", "value"]]

    # Rates from SD side-channel.
    collection_rate = _to_rm(inputs.get("collection_rate_0_1_year"), "collection_rate", base_rm, default=0.5)
    recovery_rate = _to_rm(inputs.get("recovery_rate_0_1_year"), "recovery_rate", base_rm, default=0.4)
    recycling_yield = _to_rm(inputs.get("recycling_yield_0_1_year"), "recycling_yield", base_rm, default=0.9)

    # Scrap generation split.
    old_scrap = eol.merge(collection_rate, on=["r", "m"], how="left")
    old_scrap["collection_rate"] = old_scrap["collection_rate"].fillna(0.5).clip(0.0, 1.0)
    old_scrap["value"] = old_scrap["value"] * old_scrap["collection_rate"]
    old_scrap = old_scrap[["t", "r", "m", "j", "value"]]

    new_scrap = demand.copy()
    new_scrap["value"] = new_scrap["value"] * 0.05

    # Secondary production.
    scrap_rm = pd.concat([
        old_scrap.groupby(["t", "r", "m"], as_index=False)["value"].sum(),
        new_scrap.groupby(["t", "r", "m"], as_index=False)["value"].sum(),
    ], ignore_index=True).groupby(["t", "r", "m"], as_index=False)["value"].sum().rename(columns={"value": "scrap_total"})

    sec = scrap_rm.merge(recovery_rate, on=["r", "m"], how="left")
    sec = sec.merge(recycling_yield, on=["r", "m"], how="left")
    sec["recovery_rate"] = sec["recovery_rate"].fillna(0.4).clip(0.0, 1.0)
    sec["recycling_yield"] = sec["recycling_yield"].fillna(0.9).clip(0.0, 1.0)
    sec["value"] = sec["scrap_total"] * sec["recovery_rate"] * sec["recycling_yield"]
    secondary = sec[["t", "r", "m", "value"]].copy()

    # Primary production from refining-primary capacity.
    cap = inputs.get("capacity_stage_to_sd_kt_per_yr_year")
    util = float(sd_cfg.get("capacity", {}).get("utilization_target", 0.85))
    if cap is None or cap.empty:
        primary = base_rm.copy()
        primary["t"] = year
        primary["value"] = 0.0
    else:
        c = cap.copy()
        if "p" in c.columns:
            c = c[c["p"] == "refining_primary"].copy()
        if c.empty:
            c = cap.groupby(["t", "r", "m"], as_index=False)["value"].sum()
        else:
            c = c.groupby(["t", "r", "m"], as_index=False)["value"].sum()
        primary = c.copy()
        primary["value"] = primary["value"] * util

    # Previous commodity stocks and trade.
    stock_prev = inputs.get("last_stock_kt")
    if stock_prev is None or stock_prev.empty:
        stock_prev = base_rmc.copy()
        stock_prev["value"] = 0.0

    imp_prev = inputs.get("last_imports_kt_per_yr")
    exp_prev = inputs.get("last_exports_kt_per_yr")
    if imp_prev is None or imp_prev.empty:
        imp_prev = base_rmc.copy()
        imp_prev["value"] = 0.0
    if exp_prev is None or exp_prev.empty:
        exp_prev = base_rmc.copy()
        exp_prev["value"] = 0.0

    imp_prev = imp_prev.groupby(["r", "m", "c"], as_index=False)["value"].sum().rename(columns={"value": "imports_prev"})
    exp_prev = exp_prev.groupby(["r", "m", "c"], as_index=False)["value"].sum().rename(columns={"value": "exports_prev"})

    ore = _year(inputs.get("ore_mined_kt_per_yr"), year)
    if ore.empty:
        ore = inputs.get("ore_mined_kt_per_yr") if inputs.get("ore_mined_kt_per_yr") is not None else pd.DataFrame()
    if isinstance(ore, pd.DataFrame) and not ore.empty:
        if "o" in ore.columns and "r" not in ore.columns:
            ore = ore.rename(columns={"o": "r"})
        ore_rm = ore.groupby(["r", "m"], as_index=False)["value"].sum().rename(columns={"value": "ore"})
    else:
        ore_rm = base_rm.copy()
        ore_rm["ore"] = 0.0

    # Demand realization under refined-metal availability.
    dem_rm = demand.groupby(["r", "m"], as_index=False)["value"].sum().rename(columns={"value": "desired_rm"})
    sec_rm = secondary.groupby(["r", "m"], as_index=False)["value"].sum().rename(columns={"value": "secondary_rm"})
    prim_rm = primary.groupby(["r", "m"], as_index=False)["value"].sum().rename(columns={"value": "primary_rm"})

    stock_ref_prev = stock_prev[stock_prev["c"] == "refined_metal"].groupby(["r", "m"], as_index=False)["value"].sum().rename(columns={"value": "stock_ref_prev"})
    imp_ref_prev = imp_prev[imp_prev["c"] == "refined_metal"].rename(columns={"imports_prev": "imp_ref_prev"})[["r", "m", "imp_ref_prev"]]
    exp_ref_prev = exp_prev[exp_prev["c"] == "refined_metal"].rename(columns={"exports_prev": "exp_ref_prev"})[["r", "m", "exp_ref_prev"]]

    avail = dem_rm.merge(prim_rm, on=["r", "m"], how="left")
    avail = avail.merge(sec_rm, on=["r", "m"], how="left")
    avail = avail.merge(stock_ref_prev, on=["r", "m"], how="left")
    avail = avail.merge(imp_ref_prev, on=["r", "m"], how="left")
    avail = avail.merge(exp_ref_prev, on=["r", "m"], how="left")
    for c in ["primary_rm", "secondary_rm", "stock_ref_prev", "imp_ref_prev", "exp_ref_prev"]:
        avail[c] = avail[c].fillna(0.0)
    avail["available_refined"] = (avail["primary_rm"] + avail["secondary_rm"] + avail["stock_ref_prev"] + avail["imp_ref_prev"] - avail["exp_ref_prev"]).clip(lower=0.0)
    avail["realized_rm"] = np.minimum(avail["desired_rm"], avail["available_refined"])
    avail["unmet_rm"] = (avail["desired_rm"] - avail["realized_rm"]).clip(lower=0.0)

    # Allocate realized demand across end-uses by desired shares.
    shares = demand.groupby(["r", "m"], as_index=False)["value"].sum().rename(columns={"value": "desired_rm"})
    i_use = demand.merge(shares, on=["r", "m"], how="left")
    i_use = i_use.merge(avail[["r", "m", "realized_rm"]], on=["r", "m"], how="left")
    i_use["desired_rm"] = i_use["desired_rm"].replace(0, np.nan)
    i_use["share"] = (i_use["value"] / i_use["desired_rm"]).fillna(0.0)
    i_use["value"] = i_use["share"] * i_use["realized_rm"].fillna(0.0)
    i_use = i_use[["t", "r", "m", "j", "value"]]

    # In-use stock update.
    in_use_new = in_use_prev.merge(i_use, on=["r", "m", "j"], how="left", suffixes=("_prev", "_in"))
    in_use_new = in_use_new.merge(eol[["r", "m", "j", "value"]].rename(columns={"value": "eol"}), on=["r", "m", "j"], how="left")
    in_use_new["value_in"] = in_use_new["value_in"].fillna(0.0)
    in_use_new["eol"] = in_use_new["eol"].fillna(0.0)
    in_use_new["value"] = (in_use_new["value_prev"] + in_use_new["value_in"] - in_use_new["eol"]).clip(lower=0.0)
    in_use_new = in_use_new[["r", "m", "j", "value"]]

    # Commodity stocks.
    stock = base_rmc.copy()
    stock = stock.merge(stock_prev.rename(columns={"value": "prev"}), on=["r", "m", "c"], how="left")
    stock = stock.merge(imp_prev, on=["r", "m", "c"], how="left")
    stock = stock.merge(exp_prev, on=["r", "m", "c"], how="left")
    stock = stock.merge(prim_rm, on=["r", "m"], how="left")
    stock = stock.merge(sec_rm, on=["r", "m"], how="left")
    stock = stock.merge(avail[["r", "m", "realized_rm"]], on=["r", "m"], how="left")
    stock = stock.merge(ore_rm, on=["r", "m"], how="left")

    for c in ["prev", "imports_prev", "exports_prev", "primary_rm", "secondary_rm", "realized_rm", "ore"]:
        stock[c] = stock[c].fillna(0.0)

    stock["inflow"] = 0.0
    stock["outflow"] = 0.0
    is_ref = stock["c"] == "refined_metal"
    is_scr = stock["c"] == "scrap"
    is_con = stock["c"] == "concentrate"

    scrap_total_rm = scrap_rm[["r", "m", "scrap_total"]].copy()
    stock = stock.merge(scrap_total_rm, on=["r", "m"], how="left")
    stock["scrap_total"] = stock["scrap_total"].fillna(0.0)

    # Scrap input required for secondary production.
    sec_input = sec[["r", "m", "value", "recycling_yield"]].copy()
    sec_input["sec_input"] = sec_input["value"] / np.maximum(sec_input["recycling_yield"], 1e-12)
    sec_input = sec_input[["r", "m", "sec_input"]]
    stock = stock.merge(sec_input, on=["r", "m"], how="left")
    stock["sec_input"] = stock["sec_input"].fillna(0.0)

    stock.loc[is_ref, "inflow"] = stock.loc[is_ref, "primary_rm"] + stock.loc[is_ref, "secondary_rm"] + stock.loc[is_ref, "imports_prev"]
    stock.loc[is_ref, "outflow"] = stock.loc[is_ref, "realized_rm"] + stock.loc[is_ref, "exports_prev"]

    stock.loc[is_scr, "inflow"] = stock.loc[is_scr, "scrap_total"] + stock.loc[is_scr, "imports_prev"]
    stock.loc[is_scr, "outflow"] = stock.loc[is_scr, "sec_input"] + stock.loc[is_scr, "exports_prev"]

    stock.loc[is_con, "inflow"] = stock.loc[is_con, "ore"] + stock.loc[is_con, "imports_prev"]
    stock.loc[is_con, "outflow"] = stock.loc[is_con, "primary_rm"] + stock.loc[is_con, "exports_prev"]

    stock["raw"] = stock["prev"] + stock["inflow"] - stock["outflow"]
    stock["neg_clip"] = (-stock["raw"]).clip(lower=0.0)
    stock["value"] = stock["raw"].clip(lower=0.0)
    stock["stock_change"] = stock["value"] - stock["prev"]

    stock_out = stock[["r", "m", "c", "value"]].copy()
    stock_out["t"] = year
    stock_out = stock_out[["t", "r", "m", "c", "value"]]

    stock_change = stock[["r", "m", "c", "stock_change"]].rename(columns={"stock_change": "value"})
    stock_change["t"] = year
    stock_change = stock_change[["t", "r", "m", "c", "value"]]

    apparent = stock[["r", "m", "c", "inflow"]].rename(columns={"inflow": "value"})
    apparent["t"] = year
    apparent = apparent[["t", "r", "m", "c", "value"]]

    balance = base_rmc.copy()
    balance["t"] = year
    balance["value"] = 0.0

    neg = stock[["r", "m", "c", "neg_clip"]].rename(columns={"neg_clip": "value"})
    neg["t"] = year
    neg = neg[["t", "r", "m", "c", "value"]]

    scrap_buffer = stock_out[stock_out["c"] == "scrap"][ ["t", "r", "m", "value"] ].copy()
    scrap_release = sec_input.copy()
    scrap_release["t"] = year
    scrap_release = scrap_release.rename(columns={"sec_input": "value"})[["t", "r", "m", "value"]]

    # Export caps for OD allocator (conservative fractions of available stocks).
    exp_cap = stock_out.copy()
    exp_cap["value"] = np.where(
        exp_cap["c"] == "refined_metal", exp_cap["value"] * 0.10,
        np.where(exp_cap["c"] == "scrap", exp_cap["value"] * 0.20, exp_cap["value"] * 0.20),
    )

    # Primary/secondary summaries.
    primary["t"] = year
    secondary["t"] = year

    return DMFAOutputs(
        i_use_kt_per_yr=i_use[["t", "r", "m", "j", "value"]],
        eol_outflow_kt_per_yr=eol[["t", "r", "m", "j", "value"]],
        stock_kt=stock_out,
        stock_change_kt_per_yr=stock_change,
        apparent_consumption_kt_per_yr=apparent,
        balancing_item_kt_per_yr=balance,
        negativity_clipped_kt_per_yr=neg,
        new_scrap_generated_kt_per_yr=new_scrap[["t", "r", "m", "j", "value"]],
        old_scrap_generated_kt_per_yr=old_scrap[["t", "r", "m", "j", "value"]],
        scrap_buffer_kt=scrap_buffer,
        scrap_release_kt_per_yr=scrap_release,
        export_cap_kt_per_yr=exp_cap[["t", "r", "m", "c", "value"]],
        primary_production_kt_per_yr=primary[["t", "r", "m", "value"]],
        secondary_production_kt_per_yr=secondary[["t", "r", "m", "value"]],
        eol_recycling_kt_per_yr=secondary[["t", "r", "m", "value"]],
    )
