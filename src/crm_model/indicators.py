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
