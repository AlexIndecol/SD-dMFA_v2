from __future__ import annotations
from itertools import product
from pathlib import Path
from typing import Any
import shutil

import pandas as pd
import yaml

from .config import get_paths, load_yaml
from .io_exogenous import read_all_exogenous
from .snapshot import timestamp, snapshot_configs, write_metadata
from .indicators import compute_indicators
from .coupling import build_coupled_system, run_coupled_year


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_parameters(root: Path, paths) -> dict:
    idx = load_yaml(paths.configs / "parameters.yml")
    merged: dict[str, Any] = {}
    for inc in idx.get("includes", []):
        merged = _deep_merge(merged, load_yaml(root / inc))
    return merged


def _resolve_scenario(root: Path, paths, scenario_id: str) -> dict:
    sc_index = load_yaml(paths.configs / "scenarios" / "index.yml")
    scenario_file = sc_index.get("scenarios", {}).get(scenario_id)
    if not scenario_file:
        return {"id": scenario_id, "inherits": None, "requirements": {}}

    sc = load_yaml(root / scenario_file)
    parent_id = sc.get("inherits")
    if parent_id:
        parent_file = sc_index.get("scenarios", {}).get(parent_id)
        if parent_file:
            parent = load_yaml(root / parent_file)
            sc = _deep_merge(parent, sc)
    return sc


def _build_base_grids(dim_cfg: dict) -> dict[str, pd.DataFrame]:
    dims = dim_cfg.get("dimensions", {})
    regions = dims.get("regions", {}).get("values", [])
    materials = dims.get("materials", {}).get("values", [])
    end_uses = dims.get("end_uses", {}).get("values", [])
    stages = dims.get("stages", {}).get("values", [])
    commodities = dims.get("commodities", {}).get("values", [])
    origins = dims.get("origins", {}).get("values", regions)
    destinations = dims.get("destinations", {}).get("values", regions)

    base_rmj = pd.DataFrame(product(regions, materials, end_uses), columns=["r", "m", "j"])
    base_rmp = pd.DataFrame(product(regions, materials, stages), columns=["r", "m", "p"])
    base_rmc = pd.DataFrame(product(regions, materials, commodities), columns=["r", "m", "c"])
    base_od = pd.DataFrame(product(origins, destinations, materials, commodities), columns=["o", "d", "m", "c"])

    return {
        "_base_rmj": base_rmj,
        "_base_rmp": base_rmp,
        "_base_rmc": base_rmc,
        "_base_od": base_od,
    }


def _allowed_temp_missing(assumptions_cfg: dict) -> set[str]:
    allowed: set[str] = set()
    for a in assumptions_cfg.get("assumptions", []):
        status = str(a.get("status", "")).lower()
        if status != "temp":
            continue
        for name in a.get("allows_missing_required_exogenous", []) or []:
            if isinstance(name, str) and name:
                allowed.add(name)
    return allowed


def _find_missing_required_historic(
    exo: dict[str, pd.DataFrame],
    scenario_cfg: dict,
    time_cfg: dict,
    assumptions_cfg: dict,
) -> tuple[list[str], list[str]]:
    req = scenario_cfg.get("requirements", {}).get("required_exogenous_historic", []) or []
    if not req:
        return [], []

    periods = scenario_cfg.get("periods", {}) or time_cfg.get("periods", {})
    hist = periods.get("historic", {}) or {}
    start = int(hist.get("start_year", time_cfg.get("time", {}).get("start_year", 0)))
    end = int(hist.get("end_year", time_cfg.get("time", {}).get("end_year", 0)))

    temp_ok = _allowed_temp_missing(assumptions_cfg)
    missing: list[str] = []
    missing_temp_covered: list[str] = []

    for name in req:
        df = exo.get(name)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            if name in temp_ok:
                missing_temp_covered.append(name)
            else:
                missing.append(name)
            continue

        d = df
        if "t" in d.columns:
            d = d[(d["t"] >= start) & (d["t"] <= end)]
        n_num = pd.to_numeric(d.get("value"), errors="coerce").notna().sum() if "value" in d.columns else 0
        if int(n_num) == 0:
            if name in temp_ok:
                missing_temp_covered.append(name)
            else:
                missing.append(name)

    return missing, missing_temp_covered


def _write_assumptions_used(out_dir: Path, assumptions_cfg: dict, temp_covered: list[str]) -> None:
    payload = {
        "source": "configs/assumptions.yml",
        "assumptions": assumptions_cfg.get("assumptions", []),
        "temp_allows_missing_required_exogenous_used": sorted(set(temp_covered)),
    }
    with open(out_dir / "assumptions_used.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_run_note(out_dir: Path, dry_run: bool, scenario_id: str, missing_required: list[str]) -> None:
    mode = "dry-run" if dry_run else "coupled-run"
    limits = "Required historic exogenous inputs are currently empty templates." if missing_required else "Outputs depend on configured fallback equations and input data completeness."
    text = (
        f"Decision question: Does scenario `{scenario_id}` reduce criticality/resilience risk under configured assumptions?\n"
        f"Run mode: {mode}.\n"
        f"Interpretation limits: {limits}\n"
    )
    (out_dir / "run_note.md").write_text(text, encoding="utf-8")


def _concat_series(series_parts: dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, parts in series_parts.items():
        keep = [p for p in parts if isinstance(p, pd.DataFrame) and not p.empty]
        if keep:
            out[name] = pd.concat(keep, ignore_index=True)
    return out


def run_project(root: Path, scenario_id: str, dry_run: bool = True) -> Path:
    """Run a configured project.

    - dry_run=True: ingest/snapshot and compute indicators from available series.
    - dry_run=False: execute the annual coupled fallback engine.
    """
    paths = get_paths(root)

    # Core configs
    time_cfg = load_yaml(paths.configs / "time.yml")
    coupling_cfg = load_yaml(paths.configs / "coupling.yml")
    ind_cfg = load_yaml(paths.configs / "indicators.yml")
    dim_cfg = load_yaml(paths.configs / "dimensions.yml")
    assumptions_cfg = load_yaml(paths.configs / "assumptions.yml")
    reporting_cfg = load_yaml(paths.configs / "reporting.yml")
    data_sources_cfg = load_yaml(paths.configs / "data_sources.yml")
    scenario_cfg = _resolve_scenario(root, paths, scenario_id)
    parameters_cfg = _load_parameters(root, paths)

    # Output folder + reproducibility artifacts
    out_dir = paths.outputs / scenario_id / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_configs(paths.configs, out_dir)

    # Exogenous inputs + base grids
    exo = read_all_exogenous(paths)
    exo.update(_build_base_grids(dim_cfg))

    missing_required, temp_covered = _find_missing_required_historic(exo, scenario_cfg, time_cfg, assumptions_cfg)

    if (not dry_run) and missing_required:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise ValueError(
            "Stop-the-run: required historic inputs are missing/empty and not TEMP-approved in configs/assumptions.yml: "
            + ", ".join(sorted(missing_required))
        )

    # Annual run (full mode) or passthrough (dry-run)
    series: dict[str, pd.DataFrame]
    if dry_run:
        series = {k: v for k, v in exo.items() if not k.startswith("_") and isinstance(v, pd.DataFrame)}
    else:
        cfg_bundle = {
            "time": time_cfg,
            "coupling": coupling_cfg,
            "parameters": parameters_cfg,
            "dimensions": dim_cfg,
            "assumptions": assumptions_cfg,
            "reporting": reporting_cfg,
            "data_sources": data_sources_cfg,
            "scenario": scenario_cfg,
        }
        state = build_coupled_system(cfg_bundle)
        years = range(
            int(time_cfg.get("time", {}).get("start_year", 0)),
            int(time_cfg.get("time", {}).get("end_year", -1)) + 1,
        )
        parts: dict[str, list[pd.DataFrame]] = {}
        for year in years:
            annual = run_coupled_year(state, exo, year, cfg_bundle)
            for name, df in annual.items():
                if isinstance(df, pd.DataFrame):
                    parts.setdefault(name, []).append(df)
        series = _concat_series(parts)

    # Include exogenous series alongside generated endogenous series for indicator calculations.
    exo_visible = {k: v for k, v in exo.items() if not k.startswith("_") and isinstance(v, pd.DataFrame)}
    series_all = {**exo_visible, **series}

    # Indicators
    indicator_inputs = {k: v for k, v in series_all.items() if isinstance(v, pd.DataFrame)}
    indicators = compute_indicators(ind_cfg, indicator_inputs)

    ind_dir = out_dir / "indicators"
    ind_dir.mkdir(parents=True, exist_ok=True)
    for name, df in indicators.items():
        df.to_csv(ind_dir / f"{name}.csv", index=False)

    # Series snapshots
    snap_dir = out_dir / "series_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for k, df in series_all.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(snap_dir / f"{k}.csv", index=False)

    # Run artifacts
    _write_assumptions_used(out_dir, assumptions_cfg, temp_covered)
    _write_run_note(out_dir, dry_run=dry_run, scenario_id=scenario_id, missing_required=missing_required)

    write_metadata(
        out_dir,
        {
            "scenario_id": scenario_id,
            "dry_run": dry_run,
            "time": time_cfg.get("time", {}),
            "periods": time_cfg.get("periods", {}),
            "missing_required_historic_inputs": sorted(missing_required),
            "temp_covered_missing_required_historic_inputs": sorted(temp_covered),
            "series_written": sorted(series_all.keys()),
            "note": "v4.8: annual coupled fallback engine implemented (SD+dMFA+OD trade) with stop-the-run checks for required historic inputs.",
        },
    )

    return out_dir
