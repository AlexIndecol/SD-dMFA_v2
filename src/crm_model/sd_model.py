from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class SDOutputs:
    demand_kt_per_yr: Any  # long-form dataframe: t,r,m,j,value
    capacity_stage_raw_kt_per_yr: Any  # t,r,m,p,value
    lifetime_multiplier_ge_1: Any  # t,r,m,j,value
    price_index_rel: Optional[Any] = None

def build_sd_model(configs: Dict[str, Any]) -> Any:
    """Build SD model (BPTK-Py) from configuration.
    This is a skeleton: it defines the interface and expected outputs.
    Implementation should:
      - read elasticities, adjustment times, and price dynamics settings from configs
      - use smoothed ceilings (from coupling stabilizer) for stable updates
    """
    try:
        import bptk_py  # type: ignore
    except Exception:
        bptk_py = None
    return {"engine": "bptk-py", "available": bptk_py is not None, "configs": configs}

def run_sd_step(model: Any, inputs: Dict[str, Any], year: int) -> SDOutputs:
    """Run one SD timestep.
    Inputs can include smoothed ceilings from DMFA/trade (e.g., capacity_stage_to_sd_kt_per_yr, s_sec_max_to_sd_kt_per_yr).
    Outputs must conform to SDOutputs dataclass.
    """
    raise NotImplementedError("SD step not implemented. Use BPTK-Py model equations + configs to compute outputs.")
