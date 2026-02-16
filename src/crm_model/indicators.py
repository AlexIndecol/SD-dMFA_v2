from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

EPS = 1e-12

def _ensure_df(x: Any) -> Optional[pd.DataFrame]:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError(f"Expected pandas.DataFrame, got {type(x)}")

def _sum_over(df: pd.DataFrame, dims: List[str], drop_dim: str) -> pd.DataFrame:
    keys = [d for d in dims if d != drop_dim and d in df.columns]
    out = df.groupby(keys, as_index=False)["value"].sum()
    return out

def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    return n / (d.replace(0, np.nan) + EPS)


def _normalize_region_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "r" not in out.columns and "o" in out.columns:
        out = out.rename(columns={"o": "r"})
    return out


def _group_sum(df: pd.DataFrame, keys: List[str], value_name: str) -> pd.DataFrame:
    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.groupby(keys, as_index=False)["value"].sum(min_count=1).rename(columns={"value": value_name})
    return out


def _resolve_stockpile_proxy(
    stockpile_mass: Optional[pd.DataFrame],
    stock_refined_metal: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Resolve stockpile series with fallback to refined-metal buffer stock.

    TEMP assumption: if exogenous stockpile_mass_kt is missing/empty, use
    stock_refined_metal_kt as a proxy for stockpile_kt.
    """
    if stockpile_mass is not None and not stockpile_mass.empty and "value" in stockpile_mass.columns:
        s = stockpile_mass.copy()
        keys = [c for c in ["t", "r", "m"] if c in s.columns]
        if {"r", "m"}.issubset(keys):
            return _group_sum(s, keys, "value")

    if stock_refined_metal is None or stock_refined_metal.empty or "value" not in stock_refined_metal.columns:
        return None
    s = stock_refined_metal.copy()
    if "c" in s.columns:
        s = s[s["c"] == "refined_metal"].copy()
    if s.empty:
        return None
    keys = [c for c in ["t", "r", "m"] if c in s.columns]
    if not {"r", "m"}.issubset(keys):
        return None
    return _group_sum(s, keys, "value")


def compute_indicators(ind_cfg: dict, series: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Compute a small, robust subset of indicators with explicit implementations.
    - For indicators without an implementation or missing inputs, outputs a note CSV (no silent drops).
    - Uses long-form dataframes: columns are dimension codes + 'value'.
    """
    results: Dict[str, pd.DataFrame] = {}
    indicators = ind_cfg.get("indicators", [])

    # Convenience accessors
    def get(name: str) -> Optional[pd.DataFrame]:
        return _ensure_df(series.get(name))

    for ind in indicators:
        name = ind["name"]
        req = ind.get("requires", []) or []

        if name == "apparent_consumption_kt_per_yr":
            # Preferred source: coupled-run direct accounting series.
            apparent_direct = get("apparent_consumption_kt_per_yr")
            if apparent_direct is not None and not apparent_direct.empty and "value" in apparent_direct.columns:
                ad = _normalize_region_col(apparent_direct)
                keys = [c for c in ["t", "r", "m", "c"] if c in ad.columns]
                if not {"r", "m"}.issubset(keys):
                    results[name] = pd.DataFrame({"note": ["not computed; apparent_consumption_kt_per_yr missing r/m dims"]})
                else:
                    out = _group_sum(ad, keys, "value")
                    results[name] = out[keys + ["value"]]
                continue

            domestic = get("domestic_production_kt_per_yr")
            imports = get("imports_kt_per_yr")
            exports = get("exports_kt_per_yr")
            stock_change = get("stock_change_kt_per_yr")
            if domestic is None or imports is None or exports is None or stock_change is None:
                results[name] = pd.DataFrame(
                    {"note": ["not computed; missing required inputs: ['domestic_production_kt_per_yr', 'imports_kt_per_yr', 'exports_kt_per_yr', 'stock_change_kt_per_yr']"]}
                )
                continue

            dom = _normalize_region_col(domestic.copy())
            imp = _normalize_region_col(imports.copy())
            exp = _normalize_region_col(exports.copy())
            stk = _normalize_region_col(stock_change.copy())
            if "c" not in dom.columns:
                dom["c"] = "refined_metal"

            dom_keys = [c for c in ["t", "r", "m", "c"] if c in dom.columns]
            imp_keys = [c for c in ["t", "r", "m", "c"] if c in imp.columns]
            exp_keys = [c for c in ["t", "r", "m", "c"] if c in exp.columns]
            stk_keys = [c for c in ["t", "r", "m", "c"] if c in stk.columns]
            if not {"r", "m"}.issubset(dom_keys) or not {"r", "m"}.issubset(imp_keys) or not {"r", "m"}.issubset(exp_keys) or not {"r", "m"}.issubset(stk_keys):
                results[name] = pd.DataFrame({"note": ["not computed; apparent-consumption inputs missing compatible r/m dims"]})
                continue

            keys = [c for c in ["t", "r", "m", "c"] if c in dom_keys and c in imp_keys and c in exp_keys and c in stk_keys]
            if not {"r", "m"}.issubset(keys):
                keys = [c for c in ["t", "r", "m"] if c in dom_keys and c in imp_keys and c in exp_keys and c in stk_keys]
            d = _group_sum(dom, keys, "domestic")
            i = _group_sum(imp, keys, "imports")
            e = _group_sum(exp, keys, "exports")
            s = _group_sum(stk, keys, "stock_change")
            mrg = d.merge(i, on=keys, how="outer")
            mrg = mrg.merge(e, on=keys, how="outer")
            mrg = mrg.merge(s, on=keys, how="outer")
            for c in ["domestic", "imports", "exports", "stock_change"]:
                mrg[c] = pd.to_numeric(mrg[c], errors="coerce").fillna(0.0)
            mrg["value"] = mrg["domestic"] + mrg["imports"] - mrg["exports"] + mrg["stock_change"]
            results[name] = mrg[keys + ["value"]]
            continue

        if name == "stockpile_kt":
            stockpile = _resolve_stockpile_proxy(get("stockpile_mass_kt"), get("stock_refined_metal_kt"))
            if stockpile is None or stockpile.empty:
                results[name] = pd.DataFrame({"note": ["not computed; missing required inputs: ['stockpile_mass_kt']"]})
            else:
                keys = [c for c in ["t", "r", "m"] if c in stockpile.columns]
                results[name] = stockpile[keys + ["value"]]
            continue

        if name == "stockpile_cover_years":
            stockpile = _resolve_stockpile_proxy(get("stockpile_mass_kt"), get("stock_refined_metal_kt"))
            demand = get("demand_kt_per_yr")
            if stockpile is None or stockpile.empty or demand is None or demand.empty:
                results[name] = pd.DataFrame({"note": ["not computed; missing required inputs: ['stockpile_kt', 'demand_kt_per_yr']"]})
                continue

            d = _normalize_region_col(demand.copy())
            d_keys = [c for c in ["t", "r", "m"] if c in d.columns]
            if not {"r", "m"}.issubset(d_keys):
                results[name] = pd.DataFrame({"note": ["not computed; demand_kt_per_yr missing r/m dims"]})
                continue
            d_agg = _group_sum(d, d_keys, "demand")
            s_keys = [c for c in ["t", "r", "m"] if c in stockpile.columns]
            keys = [c for c in ["t", "r", "m"] if c in d_keys and c in s_keys]
            mrg = stockpile.merge(d_agg, on=keys, how="inner")
            mrg["value"] = _safe_div(mrg["value"], mrg["demand"])
            results[name] = mrg[keys + ["value"]]
            continue

        missing = [r for r in req if r not in series]
        if missing:
            results[name] = pd.DataFrame({"note":[f"not computed; missing required inputs: {missing}"]})
            continue

        # Implemented add-ons / headline subset
        if name == "balancing_item_rel_frac":
            bal = get("balancing_item_kt_per_yr")
            ac = get("apparent_consumption_kt_per_yr")
            if bal is None or ac is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            keys = [c for c in ["t","r","m","c"] if c in bal.columns and c in ac.columns]
            mrg = bal.merge(ac, on=keys, suffixes=("_bal","_ac"), how="inner")
            mrg["value"] = _safe_div(mrg["value_bal"], mrg["value_ac"])
            results[name] = mrg[keys + ["value"]]
            continue

        if name == "buffer_coverage_years":
            stock = get("stock_refined_metal_kt")
            iuse = get("i_use_kt_per_yr")
            if stock is None or iuse is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            # stock dims: t,r,m,c; filter refined_metal if c exists
            if "c" in stock.columns:
                stock = stock[stock["c"]=="refined_metal"].copy()
            # realized demand = sum_j i_use
            if "j" in iuse.columns:
                iuse_rm = iuse.groupby(["t","r","m"], as_index=False)["value"].sum()
            else:
                iuse_rm = iuse.copy()
            st = stock.groupby(["t","r","m"], as_index=False)["value"].sum().rename(columns={"value":"stock"})
            mrg = st.merge(iuse_rm.rename(columns={"value":"iuse"}), on=["t","r","m"], how="inner")
            mrg["value"] = _safe_div(mrg["stock"], mrg["iuse"])
            results[name] = mrg[["t","r","m","value"]]
            continue

        if name == "dir_realized_frac":
            imp = get("imports_kt_per_yr")
            iuse = get("i_use_kt_per_yr")
            if imp is None or iuse is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            # imports summed over suppliers already; ensure dims t,r,m
            if "c" in imp.columns:
                # default to refined_metal for DIR unless user specifies otherwise
                imp2 = imp[imp["c"]=="refined_metal"].copy()
            else:
                imp2 = imp.copy()
            imp_rm = imp2.groupby(["t","r","m"], as_index=False)["value"].sum().rename(columns={"value":"imports"})
            iuse_rm = iuse.groupby(["t","r","m"], as_index=False)["value"].sum().rename(columns={"value":"iuse"})
            mrg = imp_rm.merge(iuse_rm, on=["t","r","m"], how="inner")
            mrg["value"] = _safe_div(mrg["imports"], mrg["iuse"])
            results[name] = mrg[["t","r","m","value"]]
            continue

        if name == "utilization_rate_ur_frac":
            iuse = get("i_use_kt_per_yr")
            prim = get("primary_production_kt_per_yr")
            sec = get("secondary_production_kt_per_yr")
            if iuse is None or prim is None or sec is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            iuse_rm = iuse.groupby(["t","r","m"], as_index=False)["value"].sum().rename(columns={"value":"iuse"})
            mrg = prim.merge(sec, on=["t","r","m"], suffixes=("_prim","_sec"), how="inner")
            mrg = mrg.merge(iuse_rm, on=["t","r","m"], how="inner")
            mrg["sourcing"] = mrg["value_prim"] + mrg["value_sec"]
            mrg["value"] = _safe_div(mrg["iuse"], mrg["sourcing"])
            results[name] = mrg[["t","r","m","value"]]
            continue

        if name == "resource_depletion_time_years":
            resources = get("resources_kt")
            primary = get("primary_production_kt_per_yr")
            if resources is None or primary is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue

            resources = _normalize_region_col(resources)
            primary = _normalize_region_col(primary)
            if "value" not in resources.columns or "value" not in primary.columns:
                results[name] = pd.DataFrame({"note":["not computed; missing value column"]})
                continue

            out_keys = [c for c in ["t", "r", "m"] if c in primary.columns]
            if not {"r", "m"}.issubset(out_keys):
                results[name] = pd.DataFrame({"note":["not computed; primary_production_kt_per_yr missing r/m dims"]})
                continue
            res_keys = [c for c in ["t", "r", "m"] if c in resources.columns and c in out_keys]
            if not {"r", "m"}.issubset(res_keys):
                results[name] = pd.DataFrame({"note":["not computed; resources_kt missing compatible r/m dims"]})
                continue

            primv = _group_sum(primary, out_keys, "primary")
            resv = _group_sum(resources, res_keys, "resources")
            mrg = primv.merge(resv, on=res_keys, how="left")
            mrg["value"] = _safe_div(mrg["resources"], mrg["primary"])
            results[name] = mrg[out_keys + ["value"]]
            continue

        if name == "hhi_generic_0_1":
            share_i = get("share_i_frac")
            if share_i is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            share_i = _normalize_region_col(share_i)
            keys = [c for c in ["t", "r", "m"] if c in share_i.columns]
            if not {"r", "m"}.issubset(keys):
                results[name] = pd.DataFrame({"note":["not computed; share_i_frac missing r/m dims"]})
                continue
            s = share_i.copy()
            s["value"] = pd.to_numeric(s["value"], errors="coerce").fillna(0.0)
            s["value"] = s["value"].pow(2)
            out = s.groupby(keys, as_index=False)["value"].sum()
            out["value"] = out["value"].clip(lower=0.0, upper=1.0)
            results[name] = out[keys + ["value"]]
            continue

        if name == "eol_rr_frac":
            old_gen = get("old_scrap_generated_kt_per_yr")
            old_rec = get("old_scrap_recycled_kt_per_yr")
            if old_gen is None or old_rec is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            old_gen = _normalize_region_col(old_gen)
            old_rec = _normalize_region_col(old_rec)
            gen_keys = [c for c in ["t", "r", "m"] if c in old_gen.columns]
            rec_keys = [c for c in ["t", "r", "m"] if c in old_rec.columns]
            if not {"r", "m"}.issubset(gen_keys) or not {"r", "m"}.issubset(rec_keys):
                results[name] = pd.DataFrame({"note":["not computed; old scrap inputs missing compatible r/m dims"]})
                continue

            gen = _group_sum(old_gen, gen_keys, "old_generated")
            rec = _group_sum(old_rec, rec_keys, "old_recycled")
            merge_keys = [c for c in rec_keys if c in gen_keys]
            mrg = gen.merge(rec, on=merge_keys, how="left")
            mrg["value"] = _safe_div(mrg["old_recycled"], mrg["old_generated"])
            results[name] = mrg[gen_keys + ["value"]]
            continue

        if name == "eol_rir_frac":
            primary_in = get("primary_input_kt_per_yr")
            sec_old = get("secondary_input_old_scrap_kt_per_yr")
            sec_tot = get("secondary_input_total_kt_per_yr")
            if primary_in is None or sec_old is None or sec_tot is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            primary_in = _normalize_region_col(primary_in)
            sec_old = _normalize_region_col(sec_old)
            sec_tot = _normalize_region_col(sec_tot)

            p_keys = [c for c in ["t", "r", "m"] if c in primary_in.columns]
            so_keys = [c for c in ["t", "r", "m"] if c in sec_old.columns]
            st_keys = [c for c in ["t", "r", "m"] if c in sec_tot.columns]
            if not {"r", "m"}.issubset(p_keys) or not {"r", "m"}.issubset(so_keys) or not {"r", "m"}.issubset(st_keys):
                results[name] = pd.DataFrame({"note":["not computed; eol_rir inputs missing compatible r/m dims"]})
                continue

            p = _group_sum(primary_in, p_keys, "primary_input")
            so = _group_sum(sec_old, so_keys, "secondary_old")
            st = _group_sum(sec_tot, st_keys, "secondary_total")
            mrg = p.merge(so, on=[c for c in so_keys if c in p_keys], how="left")
            mrg = mrg.merge(st, on=[c for c in st_keys if c in p_keys], how="left")
            mrg["value"] = _safe_div(mrg["secondary_old"], (mrg["primary_input"] + mrg["secondary_total"]))
            results[name] = mrg[p_keys + ["value"]]
            continue

        # Basic ones from INDICATORS.md that are easy
        if name == "import_reliance_frac":
            imp = get("imports_kt_per_yr")
            ship = get("shipments_kt_per_yr") or get("apparent_consumption_kt_per_yr")
            if imp is None or ship is None:
                results[name] = pd.DataFrame({"note":["not computed; missing df"]})
                continue
            # Default to refined_metal if commodity dimension present
            if "c" in imp.columns:
                imp = imp[imp["c"]=="refined_metal"].copy()
            if "c" in ship.columns:
                ship = ship[ship["c"]=="refined_metal"].copy()
            keys = [c for c in ["t","r","m"] if c in imp.columns and c in ship.columns]
            impv = imp.groupby(keys, as_index=False)["value"].sum().rename(columns={"value":"imports"})
            shv = ship.groupby(keys, as_index=False)["value"].sum().rename(columns={"value":"ship"})
            mrg = impv.merge(shv, on=keys, how="inner")
            mrg["value"] = _safe_div(mrg["imports"], mrg["ship"])
            results[name] = mrg[keys + ["value"]]
            continue

        # Default: not implemented
        formula = ind.get("formula")
        results[name] = pd.DataFrame({"note":[f"not computed; formula not implemented in engine: {formula}"]})

    return results
