from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd

from .sd_model import build_sd_model, run_sd_step, SDOutputs
from .dmfa_model import build_dmfa_model, run_dmfa_step, DMFAOutputs
from .trade_allocator import allocate_od_weights, TradeOutputs

def smooth_exponential(prev: pd.DataFrame, raw: pd.DataFrame, tau_years: float, keys: list[str]) -> pd.DataFrame:
    """First-order exponential smoothing for long-form series.
    value_sm = value_prev + (raw - value_prev)/tau
    Assumes prev and raw share keys and contain 'value'. Missing prev treated as raw.
    """
    if prev is None or prev.empty:
        return raw.copy()
    p = prev.rename(columns={"value":"prev"})
    r = raw.rename(columns={"value":"raw"})
    m = r.merge(p, on=keys, how="left")
    m["prev"] = m["prev"].fillna(m["raw"])
    m["value"] = m["prev"] + (m["raw"] - m["prev"]) / max(tau_years, 1e-6)
    return m[keys + ["value"]]

@dataclass
class CoupledState:
    sd_model: Any
    dmfa_model: Any
    ceilings: Dict[str, pd.DataFrame]

def build_coupled_system(configs: Dict[str, Any]) -> CoupledState:
    return CoupledState(
        sd_model=build_sd_model(configs),
        dmfa_model=build_dmfa_model(configs),
        ceilings={}
    )

def run_coupled_year(state: CoupledState, exogenous: Dict[str, pd.DataFrame], year: int, configs: Dict[str, Any]) -> Dict[str, Any]:
    """One annual coupled step (skeleton).
    Real implementation should:
      1) SD step (demand, capacity, lifetime multipliers)
      2) Stabilize ceilings (common tau)
      3) DMFA step (mass balance, buffers, I_use, export caps)
      4) OD trade allocation (weights) for commodities
      5) Return series dict for indicators/output
    """
    raise NotImplementedError("Coupled run not implemented; this is the interface skeleton.")
