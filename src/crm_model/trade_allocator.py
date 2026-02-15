from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class TradeOutputs:
    od_trade_flow_kt_per_yr: pd.DataFrame  # t,o,d,m,c,value
    imports_kt_per_yr: pd.DataFrame        # t,r,m,c,value
    exports_kt_per_yr: pd.DataFrame        # t,r,m,c,value

def allocate_od_weights(
    year: int,
    od_pref: pd.DataFrame,
    export_cap: pd.DataFrame,
    trade_factor: Optional[pd.DataFrame] = None,
    destination_cap: Optional[pd.DataFrame] = None,
    eps: float = 1e-12
) -> TradeOutputs:
    """Simple, robust OD allocation with weights.
    - od_pref: columns [t,o,d,m,c,value] where value is weight >=0
    - export_cap: columns [t,r,m,c,value] export availability per origin (r=o)
    - trade_factor: optional friction multiplier >=1 on [t,o,d,m,c,value]
    - destination_cap: optional ceiling per destination [t,r,m,c,value] (r=d)
    Returns OD flows and derived imports/exports.
    """
    # Filter year
    od = od_pref[od_pref["t"] == year].copy()
    ex = export_cap[export_cap["t"] == year].copy()
    if trade_factor is not None:
        tf = trade_factor[trade_factor["t"] == year].copy()
        od = od.merge(tf.rename(columns={"value":"tf"}), on=["t","o","d","m","c"], how="left")
        od["tf"] = od["tf"].fillna(1.0)
        od["w"] = od["value"] / np.maximum(od["tf"], eps)
    else:
        od["w"] = od["value"]

    # Attach export caps
    exo = ex.rename(columns={"r":"o", "value":"export_cap"})
    od = od.merge(exo[["t","o","m","c","export_cap"]], on=["t","o","m","c"], how="left")
    od["export_cap"] = od["export_cap"].fillna(0.0)

    # Normalize per origin/material/commodity
    grp = ["t","o","m","c"]
    od["w_pos"] = od["w"].clip(lower=0)
    sumw = od.groupby(grp)["w_pos"].transform("sum")
    od["share"] = np.where(sumw > 0, od["w_pos"] / sumw, 0.0)

    od["flow0"] = od["export_cap"] * od["share"]

    # Destination cap scaling (single-pass)
    if destination_cap is not None:
        dc = destination_cap[destination_cap["t"] == year].copy().rename(columns={"r":"d","value":"dest_cap"})
        od = od.merge(dc[["t","d","m","c","dest_cap"]], on=["t","d","m","c"], how="left")
        od["dest_cap"] = od["dest_cap"].fillna(np.inf)
        # implied imports by destination
        imp = od.groupby(["t","d","m","c"])["flow0"].sum().reset_index().rename(columns={"flow0":"imp0"})
        od = od.merge(imp, on=["t","d","m","c"], how="left")
        od["alpha_d"] = np.minimum(1.0, od["dest_cap"] / (od["imp0"].replace(0, np.nan) + eps))
        od["flow"] = od["flow0"] * od["alpha_d"]
    else:
        od["flow"] = od["flow0"]

    flows = od[["t","o","d","m","c"]].copy()
    flows["value"] = od["flow"]

    # Imports/exports
    exports = flows.groupby(["t","o","m","c"])["value"].sum().reset_index().rename(columns={"o":"r"})
    imports = flows.groupby(["t","d","m","c"])["value"].sum().reset_index().rename(columns={"d":"r"})

    return TradeOutputs(
        od_trade_flow_kt_per_yr=flows,
        imports_kt_per_yr=imports,
        exports_kt_per_yr=exports
    )
