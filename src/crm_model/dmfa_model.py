from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


_EXECUTION_MODE = "native_required"
_REQUIRED_STAGE_NAMES = (
    "primary_extraction",
    "beneficiation_concentration",
    "refining_primary",
    "fabrication_and_manufacturing",
    "use_phase",
    "collection",
    "sorting_preprocessing",
    "recycling_refining_secondary",
    "residue_treatment_disposal",
    "environment",
)
_REQUIRED_COMMODITIES = ("concentrate", "refined_metal", "scrap")


@dataclass
class DMFAOutputs:
    i_use_kt_per_yr: Any  # t,r,m,j,value
    eol_outflow_kt_per_yr: Any  # t,r,m,j,value
    stock_kt: Any  # t,r,m,c,value
    stock_change_kt_per_yr: Any  # t,r,m,c,value
    apparent_consumption_kt_per_yr: Any  # t,r,m,c,value
    balancing_item_kt_per_yr: Any  # t,r,m,c,value
    negativity_clipped_kt_per_yr: Any  # t,r,m,c,value
    new_scrap_generated_kt_per_yr: Any  # t,r,m,j,value
    old_scrap_generated_kt_per_yr: Any  # t,r,m,j,value
    old_scrap_recycled_kt_per_yr: Optional[Any] = None  # t,r,m,value
    primary_input_kt_per_yr: Optional[Any] = None  # t,r,m,value
    secondary_input_old_scrap_kt_per_yr: Optional[Any] = None  # t,r,m,value
    secondary_input_total_kt_per_yr: Optional[Any] = None  # t,r,m,value
    scrap_buffer_kt: Optional[Any] = None  # t,r,m,value
    scrap_release_kt_per_yr: Optional[Any] = None  # t,r,m,value
    export_cap_kt_per_yr: Optional[Any] = None  # t,r,m,c,value
    primary_production_kt_per_yr: Optional[Any] = None  # t,r,m,value
    secondary_production_kt_per_yr: Optional[Any] = None  # t,r,m,value
    eol_recycling_kt_per_yr: Optional[Any] = None  # t,r,m,value
    process_flow_kt_per_yr: Optional[Any] = None  # t,from_process,to_process,r,m,j,c,value


def _import_flodym_module() -> Optional[Any]:
    try:
        import flodym  # type: ignore
    except Exception:
        return None
    return flodym


def _resolve_native_dmfa_config(configs: Dict[str, Any]) -> tuple[str, str]:
    coupling_cfg = configs.get("coupling", {}) or {}
    dmfa_cfg = (coupling_cfg.get("modules", {}) or {}).get("dmfa", {}) or {}
    engine_requested = str(dmfa_cfg.get("engine", "flodym")).strip().lower()
    mode_requested = str(dmfa_cfg.get("execution_mode", _EXECUTION_MODE)).strip().lower()

    if engine_requested != "flodym":
        raise ValueError(
            f"Unsupported dMFA engine '{engine_requested}'. "
            "Full native runtime requires coupling.modules.dmfa.engine='flodym'."
        )
    if mode_requested != _EXECUTION_MODE:
        raise ValueError(
            f"Unsupported dMFA execution mode '{mode_requested}'. "
            "Fallback execution mode has been removed; use execution_mode='native_required'."
        )
    return engine_requested, mode_requested


def build_dmfa_model(configs: Dict[str, Any]) -> Any:
    """Build strict native dMFA runtime contract."""
    engine_requested, execution_mode = _resolve_native_dmfa_config(configs)
    flodym_mod = _import_flodym_module()
    if flodym_mod is None:
        raise RuntimeError(
            "dMFA native engine required but flodym is unavailable. "
            "Install module 'flodym' in the active environment."
        )

    return {
        "engine": engine_requested,
        "execution_mode": execution_mode,
        "native_available": True,
        "use_native": True,
        "native_backend": flodym_mod,
        "configs": configs,
    }


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


def _build_flodym_dims(flodym_mod: Any, base_df: pd.DataFrame, letters: list[str]) -> Any:
    name_map = {
        "r": "region",
        "m": "material",
        "j": "end_use",
        "c": "commodity",
    }
    dim_list = []
    for letter in letters:
        items = pd.unique(base_df[letter]).tolist()
        dim_list.append(
            flodym_mod.Dimension(
                name=name_map.get(letter, f"dim_{letter}"),
                letter=letter,
                items=items,
            )
        )
    return flodym_mod.DimensionSet(dim_list=dim_list)


def _to_flodym_array(flodym_mod: Any, dims: Any, df: pd.DataFrame, value_col: str = "value") -> Any:
    arr = flodym_mod.FlodymArray(dims=dims, values=np.zeros(dims.shape))
    if df is None or df.empty:
        return arr
    cols = list(dims.letters)
    if not set(cols).issubset(df.columns) or value_col not in df.columns:
        return arr
    use = df[cols + [value_col]].copy()
    use[value_col] = pd.to_numeric(use[value_col], errors="coerce").fillna(0.0)
    use = use.rename(columns={value_col: "value"})
    arr.set_values_from_df(use, allow_missing_values=True, allow_extra_values=False)
    return arr


def _from_flodym_array(arr: Any) -> pd.DataFrame:
    out = arr.to_df(index=False)
    rename_map = {d.name: d.letter for d in arr.dims.dim_list}
    out = out.rename(columns=rename_map)
    cols = list(arr.dims.letters) + ["value"]
    return out[cols]


def _with_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out["t"] = year
    return out[["t"] + [c for c in out.columns if c != "t"]]


def _resolve_calibration_cfg(dmfa_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read active dMFA calibration block with legacy-key compatibility."""
    calib = dmfa_cfg.get("calibration")
    if isinstance(calib, dict) and calib:
        return calib
    legacy = dmfa_cfg.get("fallback_calibration")
    if isinstance(legacy, dict) and legacy:
        return legacy
    return {}


def _ordered_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_stage_processes(configs: Dict[str, Any]) -> list[str]:
    dims_cfg = (configs.get("dimensions", {}) or {}).get("dimensions", {}) or {}
    stages_cfg = (dims_cfg.get("stages", {}) or {}).get("values", []) or []
    stage_names = _ordered_unique([s for s in stages_cfg if str(s)])

    missing = sorted(set(_REQUIRED_STAGE_NAMES).difference(set(stage_names)))
    if missing:
        raise ValueError(
            "Full flodym MFASystem process network requires the following stage names in "
            f"configs/dimensions.yml::dimensions.stages.values: {', '.join(missing)}"
        )

    return ["sysenv"] + [s for s in stage_names if s != "sysenv"]


def _build_stage_mfa_dims(flodym_mod: Any, base_rmj: pd.DataFrame, base_rmc: pd.DataFrame) -> Any:
    r_items = _ordered_unique(base_rmj["r"].tolist())
    m_items = _ordered_unique(base_rmj["m"].tolist())
    j_items = _ordered_unique(base_rmj["j"].tolist())
    c_items = _ordered_unique(base_rmc["c"].tolist())

    if not (r_items and m_items and j_items and c_items):
        raise ValueError("Cannot build full flodym MFASystem dimensions: empty r/m/j/c members.")

    return flodym_mod.DimensionSet(
        dim_list=[
            flodym_mod.Dimension(name="region", letter="r", items=r_items),
            flodym_mod.Dimension(name="material", letter="m", items=m_items),
            flodym_mod.Dimension(name="end_use", letter="j", items=j_items),
            flodym_mod.Dimension(name="commodity", letter="c", items=c_items),
        ]
    )


def _stage_flow_definitions(flodym_mod: Any) -> list[Any]:
    F = flodym_mod.FlowDefinition
    specs = [
        (("r", "m"), "sysenv", "primary_extraction", "sysenv_to_primary_extraction"),
        (("r", "m"), "primary_extraction", "beneficiation_concentration", "primary_extraction_to_beneficiation"),
        (("r", "m"), "beneficiation_concentration", "refining_primary", "beneficiation_to_refining_primary"),
        (("r", "m"), "beneficiation_concentration", "residue_treatment_disposal", "beneficiation_to_residue"),
        (("r", "m"), "refining_primary", "fabrication_and_manufacturing", "refining_to_fabrication"),
        (("r", "m", "j"), "fabrication_and_manufacturing", "use_phase", "fabrication_to_use"),
        (("r", "m", "j"), "use_phase", "collection", "use_to_collection"),
        (("r", "m", "j"), "use_phase", "residue_treatment_disposal", "use_to_residue"),
        (("r", "m"), "collection", "sorting_preprocessing", "collection_to_sorting"),
        (("r", "m"), "sorting_preprocessing", "recycling_refining_secondary", "sorting_to_recycling_secondary"),
        (("r", "m"), "sorting_preprocessing", "residue_treatment_disposal", "sorting_to_residue"),
        (("r", "m"), "recycling_refining_secondary", "refining_primary", "recycling_secondary_to_refining"),
        (("r", "m"), "recycling_refining_secondary", "residue_treatment_disposal", "recycling_secondary_to_residue"),
        (("r", "m"), "residue_treatment_disposal", "environment", "residue_to_environment"),
        (("r", "m", "j"), "use_phase", "sysenv", "use_to_sysenv_stock_build"),
        (("r", "m", "j"), "sysenv", "use_phase", "sysenv_to_use_stock_draw"),
        (
            ("r", "m", "c"),
            "sysenv",
            "beneficiation_concentration",
            "sysenv_to_beneficiation_concentrate_import",
        ),
        (
            ("r", "m", "c"),
            "beneficiation_concentration",
            "sysenv",
            "beneficiation_to_sysenv_concentrate_export",
        ),
        (("r", "m", "c"), "sysenv", "refining_primary", "sysenv_to_refining_refined_import"),
        (("r", "m", "c"), "refining_primary", "sysenv", "refining_to_sysenv_refined_export"),
        (("r", "m", "c"), "sysenv", "sorting_preprocessing", "sysenv_to_sorting_scrap_import"),
        (("r", "m", "c"), "sorting_preprocessing", "sysenv", "sorting_to_sysenv_scrap_export"),
    ]
    return [
        F(
            dim_letters=dim_letters,
            from_process_name=from_process,
            to_process_name=to_process,
            name_override=name,
        )
        for dim_letters, from_process, to_process, name in specs
    ]


def _prepare_flow_df(df: Optional[pd.DataFrame], letters: tuple[str, ...]) -> pd.DataFrame:
    cols = list(letters) + ["value"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    if "t" in out.columns:
        out = out.drop(columns=["t"])
    if "value" not in out.columns or not set(letters).issubset(out.columns):
        return pd.DataFrame(columns=cols)
    out = out[list(letters) + ["value"]].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)
    out = out.groupby(list(letters), as_index=False)["value"].sum()
    return out


def _set_mfa_flow(mfa: Any, flow_name: str, df: Optional[pd.DataFrame]) -> None:
    flow = mfa.flows[flow_name]
    flow.values[...] = 0.0
    use = _prepare_flow_df(df, tuple(flow.dims.letters))
    if use.empty:
        return
    flow.set_values_from_df(use, allow_missing_values=True, allow_extra_values=False)


def _rm_from_rmj(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["r", "m", "value"])
    out = df.copy()
    if "t" in out.columns:
        out = out.drop(columns=["t"])
    if not {"r", "m", "value"}.issubset(out.columns):
        return pd.DataFrame(columns=["r", "m", "value"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)
    return out.groupby(["r", "m"], as_index=False)["value"].sum()


def _clip_rmj_difference(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = ["r", "m", "j", "value"]
    a1 = _prepare_flow_df(a, ("r", "m", "j"))
    b1 = _prepare_flow_df(b, ("r", "m", "j"))
    if a1.empty and b1.empty:
        return pd.DataFrame(columns=cols)
    out = a1.merge(b1, on=["r", "m", "j"], how="outer", suffixes=("_a", "_b"))
    out["value"] = (out["value_a"].fillna(0.0) - out["value_b"].fillna(0.0)).clip(lower=0.0)
    return out[cols]


def _clip_rm_difference(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = ["r", "m", "value"]
    a1 = _prepare_flow_df(a, ("r", "m"))
    b1 = _prepare_flow_df(b, ("r", "m"))
    if a1.empty and b1.empty:
        return pd.DataFrame(columns=cols)
    out = a1.merge(b1, on=["r", "m"], how="outer", suffixes=("_a", "_b"))
    out["value"] = (out["value_a"].fillna(0.0) - out["value_b"].fillna(0.0)).clip(lower=0.0)
    return out[cols]


def _sum_rm(*parts: Optional[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for part in parts:
        p = _prepare_flow_df(part, ("r", "m"))
        if not p.empty:
            frames.append(p)
    if not frames:
        return pd.DataFrame(columns=["r", "m", "value"])
    return pd.concat(frames, ignore_index=True).groupby(["r", "m"], as_index=False)["value"].sum()


def _split_trade_by_commodity(df: Optional[pd.DataFrame], commodity: str) -> pd.DataFrame:
    cols = ["r", "m", "c", "value"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = _prepare_flow_df(df, ("r", "m", "c"))
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out[out["c"] == commodity][cols].copy()


def _stock_delta_split(in_use_prev: Optional[pd.DataFrame], in_use_new: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["r", "m", "j", "value"]
    prev = _prepare_flow_df(in_use_prev, ("r", "m", "j"))
    new = _prepare_flow_df(in_use_new, ("r", "m", "j"))
    if prev.empty and new.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    merged = new.merge(prev, on=["r", "m", "j"], how="outer", suffixes=("_new", "_prev"))
    delta = merged["value_new"].fillna(0.0) - merged["value_prev"].fillna(0.0)
    build = merged[["r", "m", "j"]].copy()
    draw = merged[["r", "m", "j"]].copy()
    build["value"] = delta.clip(lower=0.0)
    draw["value"] = (-delta).clip(lower=0.0)
    return build[cols], draw[cols]


def _mfa_flows_to_long_df(mfa: Any, year: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for flow in mfa.flows.values():
        df = flow.to_df(index=False)
        rename_map = {d.name: d.letter for d in flow.dims.dim_list}
        df = df.rename(columns=rename_map)
        for col in ["r", "m", "j", "c"]:
            if col not in df.columns:
                df[col] = pd.NA
        df["t"] = year
        df["from_process"] = flow.from_process.name
        df["to_process"] = flow.to_process.name
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
        rows.append(df[["t", "from_process", "to_process", "r", "m", "j", "c", "value"]])
    if not rows:
        return pd.DataFrame(columns=["t", "from_process", "to_process", "r", "m", "j", "c", "value"])
    return pd.concat(rows, ignore_index=True)


def _run_full_stage_mfa_system(
    native_backend: Any,
    configs: Dict[str, Any],
    year: int,
    *,
    base_rmj: pd.DataFrame,
    base_rmc: pd.DataFrame,
    i_use_df: pd.DataFrame,
    eol_df: pd.DataFrame,
    old_scrap_df: pd.DataFrame,
    secondary_input_total_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    ore_rm_df: pd.DataFrame,
    imp_prev_df: pd.DataFrame,
    exp_prev_df: pd.DataFrame,
    in_use_prev_df: pd.DataFrame,
    in_use_new_df: pd.DataFrame,
) -> pd.DataFrame:
    process_names = _resolve_stage_processes(configs)
    dims = _build_stage_mfa_dims(native_backend, base_rmj=base_rmj, base_rmc=base_rmc)
    flow_defs = _stage_flow_definitions(native_backend)

    processes = native_backend.make_processes(process_names)
    flows = native_backend.make_empty_flows(processes=processes, flow_definitions=flow_defs, dims=dims)

    class CRMStageMFASystem(native_backend.MFASystem):
        def compute(self):
            ctx = self.runtime_context
            i_use = _prepare_flow_df(ctx["i_use_df"], ("r", "m", "j"))
            eol = _prepare_flow_df(ctx["eol_df"], ("r", "m", "j"))
            old_scrap = _prepare_flow_df(ctx["old_scrap_df"], ("r", "m", "j"))
            secondary_input_total = _prepare_flow_df(ctx["secondary_input_total_df"], ("r", "m"))
            secondary_prod = _prepare_flow_df(ctx["secondary_df"], ("r", "m"))
            primary_prod = _prepare_flow_df(ctx["primary_df"], ("r", "m"))
            ore_rm = _prepare_flow_df(ctx["ore_rm_df"], ("r", "m"))
            imports_rmc = _prepare_flow_df(ctx["imp_prev_df"], ("r", "m", "c"))
            exports_rmc = _prepare_flow_df(ctx["exp_prev_df"], ("r", "m", "c"))

            use_to_residue = _clip_rmj_difference(eol, old_scrap)
            old_scrap_rm = _rm_from_rmj(old_scrap)
            sorting_to_residue = _clip_rm_difference(old_scrap_rm, secondary_input_total)
            beneficiation_to_residue = _clip_rm_difference(ore_rm, primary_prod)
            recycling_to_residue = _clip_rm_difference(secondary_input_total, secondary_prod)
            residue_to_environment = _sum_rm(
                _rm_from_rmj(use_to_residue),
                sorting_to_residue,
                beneficiation_to_residue,
                recycling_to_residue,
            )
            refining_to_fabrication = _rm_from_rmj(i_use)
            stock_build, stock_draw = _stock_delta_split(ctx["in_use_prev_df"], ctx["in_use_new_df"])

            _set_mfa_flow(self, "sysenv_to_primary_extraction", ore_rm)
            _set_mfa_flow(self, "primary_extraction_to_beneficiation", ore_rm)
            _set_mfa_flow(self, "beneficiation_to_refining_primary", primary_prod)
            _set_mfa_flow(self, "beneficiation_to_residue", beneficiation_to_residue)
            _set_mfa_flow(self, "refining_to_fabrication", refining_to_fabrication)
            _set_mfa_flow(self, "fabrication_to_use", i_use)
            _set_mfa_flow(self, "use_to_collection", old_scrap)
            _set_mfa_flow(self, "use_to_residue", use_to_residue)
            _set_mfa_flow(self, "collection_to_sorting", old_scrap_rm)
            _set_mfa_flow(self, "sorting_to_recycling_secondary", secondary_input_total)
            _set_mfa_flow(self, "sorting_to_residue", sorting_to_residue)
            _set_mfa_flow(self, "recycling_secondary_to_refining", secondary_prod)
            _set_mfa_flow(self, "recycling_secondary_to_residue", recycling_to_residue)
            _set_mfa_flow(self, "residue_to_environment", residue_to_environment)
            _set_mfa_flow(self, "use_to_sysenv_stock_build", stock_build)
            _set_mfa_flow(self, "sysenv_to_use_stock_draw", stock_draw)

            _set_mfa_flow(self, "sysenv_to_beneficiation_concentrate_import", _split_trade_by_commodity(imports_rmc, "concentrate"))
            _set_mfa_flow(self, "beneficiation_to_sysenv_concentrate_export", _split_trade_by_commodity(exports_rmc, "concentrate"))
            _set_mfa_flow(self, "sysenv_to_refining_refined_import", _split_trade_by_commodity(imports_rmc, "refined_metal"))
            _set_mfa_flow(self, "refining_to_sysenv_refined_export", _split_trade_by_commodity(exports_rmc, "refined_metal"))
            _set_mfa_flow(self, "sysenv_to_sorting_scrap_import", _split_trade_by_commodity(imports_rmc, "scrap"))
            _set_mfa_flow(self, "sorting_to_sysenv_scrap_export", _split_trade_by_commodity(exports_rmc, "scrap"))

    mfa = CRMStageMFASystem(
        dims=dims,
        parameters={},
        processes=processes,
        flows=flows,
        stocks={},
        runtime_context={
            "i_use_df": i_use_df,
            "eol_df": eol_df,
            "old_scrap_df": old_scrap_df,
            "secondary_input_total_df": secondary_input_total_df,
            "secondary_df": secondary_df,
            "primary_df": primary_df,
            "ore_rm_df": ore_rm_df,
            "imp_prev_df": imp_prev_df,
            "exp_prev_df": exp_prev_df,
            "in_use_prev_df": in_use_prev_df,
            "in_use_new_df": in_use_new_df,
        },
    )
    mfa.compute()
    return _mfa_flows_to_long_df(mfa, year=year)


def _run_dmfa_step_native(model: Any, inputs: Dict[str, Any], year: int) -> DMFAOutputs:
    native_backend = model.get("native_backend")
    if native_backend is None:
        raise RuntimeError("flodym backend unavailable for native dMFA execution.")

    cfg = model.get("configs", {})
    sd_cfg = cfg.get("parameters", {}).get("sd", {}) or {}
    dmfa_cfg = cfg.get("parameters", {}).get("dmfa", {}) or {}
    calib_cfg = _resolve_calibration_cfg(dmfa_cfg)

    new_scrap_frac = float(np.clip(float(calib_cfg.get("new_scrap_fraction_of_demand", 0.05)), 0.0, 1.0))
    eol_outflow_mult = float(np.clip(float(calib_cfg.get("eol_outflow_multiplier", 1.0)), 0.01, 5.0))
    exp_cap_frac = calib_cfg.get("export_cap_fraction_by_commodity", {}) or {}
    exp_cap_ref = float(np.clip(float(exp_cap_frac.get("refined_metal", 0.10)), 0.0, 1.0))
    exp_cap_scr = float(np.clip(float(exp_cap_frac.get("scrap", 0.20)), 0.0, 1.0))
    exp_cap_con = float(np.clip(float(exp_cap_frac.get("concentrate", 0.20)), 0.0, 1.0))
    util = float(sd_cfg.get("capacity", {}).get("utilization_target", 0.85))

    base_rmj = inputs.get("base_rmj")
    base_rmc = inputs.get("base_rmc")
    if base_rmj is None or base_rmc is None or base_rmj.empty or base_rmc.empty:
        raise ValueError("DMFA step requires non-empty base grids: base_rmj and base_rmc")

    base_rm = base_rmj[["r", "m"]].drop_duplicates()
    dims_rmj = _build_flodym_dims(native_backend, base_rmj, ["r", "m", "j"])
    dims_rmc = _build_flodym_dims(native_backend, base_rmc, ["r", "m", "c"])
    dims_rm = _build_flodym_dims(native_backend, base_rm, ["r", "m"])

    c_items = list(pd.unique(base_rmc["c"]))
    c_idx = {c: i for i, c in enumerate(c_items)}
    if any(commodity not in c_idx for commodity in _REQUIRED_COMMODITIES):
        raise ValueError("Commodity dimension must include concentrate, refined_metal, and scrap.")

    demand = inputs.get("demand_kt_per_yr_year")
    if demand is None or demand.empty:
        demand = base_rmj.copy()
        demand["t"] = year
        demand["value"] = 0.0
    demand = demand[["t", "r", "m", "j", "value"]].copy()
    demand_arr = _to_flodym_array(native_backend, dims_rmj, demand.drop(columns=["t"]))

    in_use_prev = inputs.get("last_in_use_stock_kt")
    if in_use_prev is None or in_use_prev.empty:
        in_use_prev = base_rmj.copy()
        in_use_prev["value"] = 0.0
    in_use_prev_arr = _to_flodym_array(native_backend, dims_rmj, in_use_prev[["r", "m", "j", "value"]])

    life_df = _lifetime_mean(base_rmj, inputs, year)
    life_arr = _to_flodym_array(native_backend, dims_rmj, life_df[["r", "m", "j", "value"]])

    life_mult = inputs.get("lifetime_multiplier_ge_1_year")
    if life_mult is None or life_mult.empty:
        life_mult = base_rmj.copy()
        life_mult["value"] = 1.0
    life_mult = life_mult[["r", "m", "j", "value"]].copy()
    life_mult["value"] = pd.to_numeric(life_mult["value"], errors="coerce").fillna(1.0).clip(lower=1.0)
    life_mult_arr = _to_flodym_array(native_backend, dims_rmj, life_mult)

    eol_values = (in_use_prev_arr.values / np.maximum(life_arr.values * life_mult_arr.values, 1e-12)) * eol_outflow_mult
    eol_arr = native_backend.FlodymArray(dims=dims_rmj, values=eol_values, name="eol_outflow")

    collection_df = _to_rm(inputs.get("collection_rate_0_1_year"), "collection_rate", base_rm, default=0.5)
    collection_df["collection_rate"] = pd.to_numeric(collection_df["collection_rate"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    collection_arr = _to_flodym_array(
        native_backend,
        dims_rm,
        collection_df.rename(columns={"collection_rate": "value"})[["r", "m", "value"]],
    )
    recovery_df = _to_rm(inputs.get("recovery_rate_0_1_year"), "recovery_rate", base_rm, default=0.4)
    recovery_df["recovery_rate"] = pd.to_numeric(recovery_df["recovery_rate"], errors="coerce").fillna(0.4).clip(0.0, 1.0)
    recovery_arr = _to_flodym_array(
        native_backend,
        dims_rm,
        recovery_df.rename(columns={"recovery_rate": "value"})[["r", "m", "value"]],
    )
    yield_df = _to_rm(inputs.get("recycling_yield_0_1_year"), "recycling_yield", base_rm, default=0.9)
    yield_df["recycling_yield"] = pd.to_numeric(yield_df["recycling_yield"], errors="coerce").fillna(0.9).clip(0.0, 1.0)
    yield_arr = _to_flodym_array(
        native_backend,
        dims_rm,
        yield_df.rename(columns={"recycling_yield": "value"})[["r", "m", "value"]],
    )

    old_scrap_values = eol_arr.values * collection_arr.values[:, :, None]
    new_scrap_values = demand_arr.values * new_scrap_frac
    old_scrap_arr = native_backend.FlodymArray(dims=dims_rmj, values=old_scrap_values, name="old_scrap_generated")
    new_scrap_arr = native_backend.FlodymArray(dims=dims_rmj, values=new_scrap_values, name="new_scrap_generated")

    old_scrap_rm_values = old_scrap_values.sum(axis=2)
    new_scrap_rm_values = new_scrap_values.sum(axis=2)
    scrap_total_rm_values = old_scrap_rm_values + new_scrap_rm_values

    secondary_input_total_values = scrap_total_rm_values * recovery_arr.values
    secondary_values = secondary_input_total_values * yield_arr.values
    secondary_input_old_values = old_scrap_rm_values * recovery_arr.values
    old_scrap_recycled_values = secondary_input_old_values * yield_arr.values
    sec_input_values = np.divide(secondary_values, np.maximum(yield_arr.values, 1e-12))

    secondary_arr = native_backend.FlodymArray(dims=dims_rm, values=secondary_values, name="secondary_production")
    secondary_input_total_arr = native_backend.FlodymArray(dims=dims_rm, values=secondary_input_total_values, name="secondary_input_total")
    secondary_input_old_arr = native_backend.FlodymArray(dims=dims_rm, values=secondary_input_old_values, name="secondary_input_old")
    old_scrap_recycled_arr = native_backend.FlodymArray(dims=dims_rm, values=old_scrap_recycled_values, name="old_scrap_recycled")
    sec_input_arr = native_backend.FlodymArray(dims=dims_rm, values=sec_input_values, name="secondary_input_required")

    cap = inputs.get("capacity_stage_to_sd_kt_per_yr_year")
    if cap is None or cap.empty:
        primary_arr = native_backend.FlodymArray(dims=dims_rm, values=np.zeros(dims_rm.shape), name="primary_production")
    else:
        cap_work = cap.copy()
        if "p" in cap_work.columns:
            cap_ref = cap_work[cap_work["p"] == "refining_primary"].copy()
            if cap_ref.empty:
                cap_ref = cap_work.groupby(["r", "m"], as_index=False)["value"].sum()
            else:
                cap_ref = cap_ref.groupby(["r", "m"], as_index=False)["value"].sum()
        else:
            cap_ref = cap_work.groupby(["r", "m"], as_index=False)["value"].sum()
        cap_ref["value"] = pd.to_numeric(cap_ref["value"], errors="coerce").fillna(0.0) * util
        primary_arr = _to_flodym_array(native_backend, dims_rm, cap_ref[["r", "m", "value"]])

    stock_prev = inputs.get("last_stock_kt")
    if stock_prev is None or stock_prev.empty:
        stock_prev = base_rmc.copy()
        stock_prev["value"] = 0.0
    stock_prev_arr = _to_flodym_array(native_backend, dims_rmc, stock_prev[["r", "m", "c", "value"]])

    imp_prev = inputs.get("last_imports_kt_per_yr")
    if imp_prev is None or imp_prev.empty:
        imp_prev = base_rmc.copy()
        imp_prev["value"] = 0.0
    else:
        imp_prev = imp_prev.groupby(["r", "m", "c"], as_index=False)["value"].sum()
    imp_prev_arr = _to_flodym_array(native_backend, dims_rmc, imp_prev[["r", "m", "c", "value"]])

    exp_prev = inputs.get("last_exports_kt_per_yr")
    if exp_prev is None or exp_prev.empty:
        exp_prev = base_rmc.copy()
        exp_prev["value"] = 0.0
    else:
        exp_prev = exp_prev.groupby(["r", "m", "c"], as_index=False)["value"].sum()
    exp_prev_arr = _to_flodym_array(native_backend, dims_rmc, exp_prev[["r", "m", "c", "value"]])

    ore = _year(inputs.get("ore_mined_kt_per_yr"), year)
    if ore.empty:
        ore = inputs.get("ore_mined_kt_per_yr") if inputs.get("ore_mined_kt_per_yr") is not None else pd.DataFrame()
    if isinstance(ore, pd.DataFrame) and not ore.empty:
        if "o" in ore.columns and "r" not in ore.columns:
            ore = ore.rename(columns={"o": "r"})
        ore_rm = ore.groupby(["r", "m"], as_index=False)["value"].sum()
    else:
        ore_rm = base_rm.copy()
        ore_rm["value"] = 0.0
    ore_arr = _to_flodym_array(native_backend, dims_rm, ore_rm[["r", "m", "value"]])

    desired_rm_values = demand_arr.values.sum(axis=2)
    sec_rm_values = secondary_arr.values
    prim_rm_values = primary_arr.values
    ref_idx = c_idx["refined_metal"]
    scr_idx = c_idx["scrap"]
    con_idx = c_idx["concentrate"]

    stock_ref_prev = stock_prev_arr.values[:, :, ref_idx]
    imp_ref_prev = imp_prev_arr.values[:, :, ref_idx]
    exp_ref_prev = exp_prev_arr.values[:, :, ref_idx]
    available_refined = np.maximum(prim_rm_values + sec_rm_values + stock_ref_prev + imp_ref_prev - exp_ref_prev, 0.0)
    realized_rm_values = np.minimum(desired_rm_values, available_refined)

    demand_share = np.divide(
        demand_arr.values,
        desired_rm_values[:, :, None],
        out=np.zeros_like(demand_arr.values),
        where=desired_rm_values[:, :, None] > 0,
    )
    i_use_values = demand_share * realized_rm_values[:, :, None]
    i_use_arr = native_backend.FlodymArray(dims=dims_rmj, values=i_use_values, name="i_use")

    in_use_new_values = np.maximum(in_use_prev_arr.values + i_use_values - eol_values, 0.0)

    inflow_values = np.zeros(dims_rmc.shape)
    outflow_values = np.zeros(dims_rmc.shape)
    inflow_values[:, :, ref_idx] = prim_rm_values + sec_rm_values + imp_ref_prev
    outflow_values[:, :, ref_idx] = realized_rm_values + exp_ref_prev
    inflow_values[:, :, scr_idx] = scrap_total_rm_values + imp_prev_arr.values[:, :, scr_idx]
    outflow_values[:, :, scr_idx] = sec_input_arr.values + exp_prev_arr.values[:, :, scr_idx]
    inflow_values[:, :, con_idx] = ore_arr.values + imp_prev_arr.values[:, :, con_idx]
    outflow_values[:, :, con_idx] = prim_rm_values + exp_prev_arr.values[:, :, con_idx]

    raw_stock_values = stock_prev_arr.values + inflow_values - outflow_values
    neg_values = np.maximum(-raw_stock_values, 0.0)
    stock_values = np.maximum(raw_stock_values, 0.0)
    change_values = stock_values - stock_prev_arr.values

    stock_arr = native_backend.FlodymArray(dims=dims_rmc, values=stock_values, name="stock")
    stock_change_arr = native_backend.FlodymArray(dims=dims_rmc, values=change_values, name="stock_change")
    apparent_arr = native_backend.FlodymArray(dims=dims_rmc, values=inflow_values, name="apparent_consumption")
    balance_arr = native_backend.FlodymArray(dims=dims_rmc, values=np.zeros(dims_rmc.shape), name="balancing_item")
    neg_arr = native_backend.FlodymArray(dims=dims_rmc, values=neg_values, name="negativity_clipped")

    exp_cap_values = stock_values.copy()
    exp_cap_values[:, :, ref_idx] = exp_cap_values[:, :, ref_idx] * exp_cap_ref
    exp_cap_values[:, :, scr_idx] = exp_cap_values[:, :, scr_idx] * exp_cap_scr
    exp_cap_values[:, :, con_idx] = exp_cap_values[:, :, con_idx] * exp_cap_con
    exp_cap_arr = native_backend.FlodymArray(dims=dims_rmc, values=exp_cap_values, name="export_cap")

    scrap_buffer_arr = native_backend.FlodymArray(dims=dims_rm, values=stock_values[:, :, scr_idx], name="scrap_buffer")
    in_use_new_arr = native_backend.FlodymArray(dims=dims_rmj, values=in_use_new_values, name="in_use_stock")

    i_use_df = _with_year(_from_flodym_array(i_use_arr), year)
    eol_df = _with_year(_from_flodym_array(eol_arr), year)
    stock_df = _with_year(_from_flodym_array(stock_arr), year)
    stock_change_df = _with_year(_from_flodym_array(stock_change_arr), year)
    apparent_df = _with_year(_from_flodym_array(apparent_arr), year)
    balance_df = _with_year(_from_flodym_array(balance_arr), year)
    neg_df = _with_year(_from_flodym_array(neg_arr), year)
    new_scrap_df = _with_year(_from_flodym_array(new_scrap_arr), year)
    old_scrap_df = _with_year(_from_flodym_array(old_scrap_arr), year)
    old_scrap_recycled_df = _with_year(_from_flodym_array(old_scrap_recycled_arr), year)
    primary_df = _with_year(_from_flodym_array(primary_arr), year)
    secondary_input_old_df = _with_year(_from_flodym_array(secondary_input_old_arr), year)
    secondary_input_total_df = _with_year(_from_flodym_array(secondary_input_total_arr), year)
    scrap_buffer_df = _with_year(_from_flodym_array(scrap_buffer_arr), year)
    scrap_release_df = _with_year(_from_flodym_array(sec_input_arr), year)
    exp_cap_df = _with_year(_from_flodym_array(exp_cap_arr), year)
    secondary_df = _with_year(_from_flodym_array(secondary_arr), year)
    in_use_new_df = _with_year(_from_flodym_array(in_use_new_arr), year)

    process_flow_df = _run_full_stage_mfa_system(
        native_backend=native_backend,
        configs=cfg,
        year=year,
        base_rmj=base_rmj,
        base_rmc=base_rmc,
        i_use_df=i_use_df[["t", "r", "m", "j", "value"]],
        eol_df=eol_df[["t", "r", "m", "j", "value"]],
        old_scrap_df=old_scrap_df[["t", "r", "m", "j", "value"]],
        secondary_input_total_df=secondary_input_total_df[["t", "r", "m", "value"]],
        secondary_df=secondary_df[["t", "r", "m", "value"]],
        primary_df=primary_df[["t", "r", "m", "value"]],
        ore_rm_df=ore_rm[["r", "m", "value"]],
        imp_prev_df=imp_prev[["r", "m", "c", "value"]],
        exp_prev_df=exp_prev[["r", "m", "c", "value"]],
        in_use_prev_df=in_use_prev[["r", "m", "j", "value"]],
        in_use_new_df=in_use_new_df[["t", "r", "m", "j", "value"]],
    )

    return DMFAOutputs(
        i_use_kt_per_yr=i_use_df[["t", "r", "m", "j", "value"]],
        eol_outflow_kt_per_yr=eol_df[["t", "r", "m", "j", "value"]],
        stock_kt=stock_df[["t", "r", "m", "c", "value"]],
        stock_change_kt_per_yr=stock_change_df[["t", "r", "m", "c", "value"]],
        apparent_consumption_kt_per_yr=apparent_df[["t", "r", "m", "c", "value"]],
        balancing_item_kt_per_yr=balance_df[["t", "r", "m", "c", "value"]],
        negativity_clipped_kt_per_yr=neg_df[["t", "r", "m", "c", "value"]],
        new_scrap_generated_kt_per_yr=new_scrap_df[["t", "r", "m", "j", "value"]],
        old_scrap_generated_kt_per_yr=old_scrap_df[["t", "r", "m", "j", "value"]],
        old_scrap_recycled_kt_per_yr=old_scrap_recycled_df[["t", "r", "m", "value"]],
        primary_input_kt_per_yr=primary_df[["t", "r", "m", "value"]],
        secondary_input_old_scrap_kt_per_yr=secondary_input_old_df[["t", "r", "m", "value"]],
        secondary_input_total_kt_per_yr=secondary_input_total_df[["t", "r", "m", "value"]],
        scrap_buffer_kt=scrap_buffer_df[["t", "r", "m", "value"]],
        scrap_release_kt_per_yr=scrap_release_df[["t", "r", "m", "value"]],
        export_cap_kt_per_yr=exp_cap_df[["t", "r", "m", "c", "value"]],
        primary_production_kt_per_yr=primary_df[["t", "r", "m", "value"]],
        secondary_production_kt_per_yr=secondary_df[["t", "r", "m", "value"]],
        eol_recycling_kt_per_yr=secondary_df[["t", "r", "m", "value"]],
        process_flow_kt_per_yr=process_flow_df,
    )


def run_dmfa_step(model: Any, inputs: Dict[str, Any], year: int) -> DMFAOutputs:
    """Run one dMFA timestep with strict native flodym execution."""
    if model.get("engine") != "flodym" or not bool(model.get("use_native", False)):
        raise RuntimeError(
            "dMFA native runtime unavailable. Full native migration requires flodym. "
            "Ensure coupling.modules.dmfa.execution_mode='native_required' and dependency installed."
        )
    return _run_dmfa_step_native(model=model, inputs=inputs, year=year)
