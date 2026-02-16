from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from crm_model.config import get_paths, load_yaml
from crm_model.coupling import build_coupled_system, run_coupled_year
from crm_model.io_exogenous import read_all_exogenous
from crm_model.snapshot import timestamp


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


def _parse_grid(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Calibration grid cannot be empty.")
    return vals


@dataclass
class CalibrationMetrics:
    mae_i_use: float
    rmse_i_use: float
    mape_i_use: float
    mae_stock: float
    rmse_stock: float
    mape_stock: float
    objective: float


def _calc_pair_metrics(model: pd.DataFrame, obs: pd.DataFrame, keys: list[str]) -> tuple[float, float, float]:
    m = model.merge(obs, on=keys, how="inner", suffixes=("_model", "_obs"))
    if m.empty:
        return float("inf"), float("inf"), float("inf")

    m["value_model"] = pd.to_numeric(m["value_model"], errors="coerce")
    m["value_obs"] = pd.to_numeric(m["value_obs"], errors="coerce")
    m = m[m["value_obs"].notna()].copy()
    if m.empty:
        return float("inf"), float("inf"), float("inf")

    err = m["value_model"] - m["value_obs"]
    abs_err = err.abs()
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.square(err).mean()))
    positive_obs = m["value_obs"] > 0
    if positive_obs.any():
        mape = float((abs_err[positive_obs] / m.loc[positive_obs, "value_obs"]).mean())
    else:
        mape = float("inf")
    return mae, rmse, mape


def _compute_metrics(
    series_i_use: pd.DataFrame,
    series_stock: pd.DataFrame,
    obs_i_use: pd.DataFrame,
    obs_stock: pd.DataFrame,
    hist_start: int,
    hist_end: int,
    w_i_use: float,
    w_stock: float,
) -> CalibrationMetrics:
    i_use_model = series_i_use.copy()
    stock_model = series_stock.copy()
    i_use_obs = obs_i_use.copy()
    stock_obs = obs_stock.copy()

    for df in (i_use_model, stock_model, i_use_obs, stock_obs):
        df["t"] = pd.to_numeric(df["t"], errors="coerce")

    i_use_model = i_use_model[i_use_model["t"].between(hist_start, hist_end)]
    stock_model = stock_model[stock_model["t"].between(hist_start, hist_end)]
    i_use_obs = i_use_obs[i_use_obs["t"].between(hist_start, hist_end)]
    stock_obs = stock_obs[stock_obs["t"].between(hist_start, hist_end)]

    keys = ["t", "r", "m", "j"]
    mae_i, rmse_i, mape_i = _calc_pair_metrics(i_use_model, i_use_obs, keys)
    mae_s, rmse_s, mape_s = _calc_pair_metrics(stock_model, stock_obs, keys)

    objective = (w_i_use * mape_i) + (w_stock * mape_s)
    return CalibrationMetrics(
        mae_i_use=mae_i,
        rmse_i_use=rmse_i,
        mape_i_use=mape_i,
        mae_stock=mae_s,
        rmse_stock=rmse_s,
        mape_stock=mape_s,
        objective=float(objective),
    )


def _run_candidate(
    cfg_bundle: dict[str, Any],
    exogenous: dict[str, pd.DataFrame],
    run_start: int,
    run_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = build_coupled_system(cfg_bundle)
    i_use_parts: list[pd.DataFrame] = []
    stock_parts: list[pd.DataFrame] = []
    for year in range(run_start, run_end + 1):
        annual = run_coupled_year(state, exogenous, year, cfg_bundle)
        i_use_parts.append(annual["i_use_kt_per_yr"])
        stock_parts.append(annual["in_use_stock_kt"])
    return pd.concat(i_use_parts, ignore_index=True), pd.concat(stock_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline fallback calibration sweep.")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--util-grid", default="0.75,0.85,0.95,1.05")
    parser.add_argument("--eol-mult-grid", default="0.85,1.0,1.15")
    parser.add_argument("--ref-exp-grid", default="0.05,0.10,0.15")
    parser.add_argument("--new-scrap-grid", default="0.05")
    parser.add_argument("--objective-weight-i-use", type=float, default=0.5)
    parser.add_argument("--objective-weight-stock", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.objective_weight_i_use < 0 or args.objective_weight_stock < 0:
        raise ValueError("Objective weights must be non-negative.")
    if (args.objective_weight_i_use + args.objective_weight_stock) <= 0:
        raise ValueError("At least one objective weight must be > 0.")

    wsum = args.objective_weight_i_use + args.objective_weight_stock
    w_i_use = args.objective_weight_i_use / wsum
    w_stock = args.objective_weight_stock / wsum

    util_grid = _parse_grid(args.util_grid)
    eol_mult_grid = _parse_grid(args.eol_mult_grid)
    ref_exp_grid = _parse_grid(args.ref_exp_grid)
    new_scrap_grid = _parse_grid(args.new_scrap_grid)

    root = Path(__file__).resolve().parents[2]
    paths = get_paths(root)

    time_cfg = load_yaml(paths.configs / "time.yml")
    coupling_cfg = load_yaml(paths.configs / "coupling.yml")
    dim_cfg = load_yaml(paths.configs / "dimensions.yml")
    assumptions_cfg = load_yaml(paths.configs / "assumptions.yml")
    reporting_cfg = load_yaml(paths.configs / "reporting.yml")
    data_sources_cfg = load_yaml(paths.configs / "data_sources.yml")
    scenario_cfg = _resolve_scenario(root, paths, args.scenario)
    parameters_cfg = _load_parameters(root, paths)

    periods = scenario_cfg.get("periods", {}) or time_cfg.get("periods", {})
    hist = periods.get("historic", {}) or {}
    hist_start = int(hist.get("start_year", 1870))
    hist_end = int(hist.get("end_year", 2019))
    run_start = int(time_cfg.get("time", {}).get("start_year", hist_start))
    run_end = hist_end

    exo = read_all_exogenous(paths)
    exo.update(_build_base_grids(dim_cfg))

    obs_i_use = exo["gas_to_use_observed_kt_per_yr"][["t", "r", "m", "j", "value"]].copy()
    obs_stock = exo["in_use_stock_observed_kt"][["t", "r", "m", "j", "value"]].copy()

    out_dir = root / "outputs" / "calibration" / args.scenario / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(product(util_grid, eol_mult_grid, ref_exp_grid, new_scrap_grid))
    rows: list[dict[str, Any]] = []
    total = len(candidates)

    for idx, (util, eol_mult, ref_exp, new_scrap) in enumerate(candidates, start=1):
        params = deepcopy(parameters_cfg)
        params.setdefault("sd", {}).setdefault("capacity", {})["utilization_target"] = float(util)

        dcal = params.setdefault("dmfa", {}).setdefault("fallback_calibration", {})
        dcal["eol_outflow_multiplier"] = float(eol_mult)
        dcal["new_scrap_fraction_of_demand"] = float(new_scrap)
        dcal.setdefault("export_cap_fraction_by_commodity", {})
        dcal["export_cap_fraction_by_commodity"]["refined_metal"] = float(ref_exp)

        cfg_bundle = {
            "time": time_cfg,
            "coupling": coupling_cfg,
            "parameters": params,
            "dimensions": dim_cfg,
            "assumptions": assumptions_cfg,
            "reporting": reporting_cfg,
            "data_sources": data_sources_cfg,
            "scenario": scenario_cfg,
        }

        i_use_series, stock_series = _run_candidate(
            cfg_bundle=cfg_bundle,
            exogenous=exo,
            run_start=run_start,
            run_end=run_end,
        )
        metrics = _compute_metrics(
            series_i_use=i_use_series,
            series_stock=stock_series,
            obs_i_use=obs_i_use,
            obs_stock=obs_stock,
            hist_start=hist_start,
            hist_end=hist_end,
            w_i_use=w_i_use,
            w_stock=w_stock,
        )

        row = {
            "candidate_id": idx,
            "utilization_target": float(util),
            "eol_outflow_multiplier": float(eol_mult),
            "refined_export_cap_fraction": float(ref_exp),
            "new_scrap_fraction_of_demand": float(new_scrap),
            "mae_i_use": metrics.mae_i_use,
            "rmse_i_use": metrics.rmse_i_use,
            "mape_i_use": metrics.mape_i_use,
            "mae_stock": metrics.mae_stock,
            "rmse_stock": metrics.rmse_stock,
            "mape_stock": metrics.mape_stock,
            "objective_weighted_mape": metrics.objective,
        }
        rows.append(row)
        print(
            f"[{idx}/{total}] util={util:.3f} eol_mult={eol_mult:.3f} "
            f"ref_exp={ref_exp:.3f} new_scrap={new_scrap:.3f} "
            f"objective={metrics.objective:.6f}"
        )

    results = pd.DataFrame(rows).sort_values("objective_weighted_mape").reset_index(drop=True)
    top = results.head(max(int(args.top_k), 1)).copy()

    best = results.iloc[0].to_dict()
    best_params = {
        "sd.capacity.utilization_target": float(best["utilization_target"]),
        "dmfa.fallback_calibration.eol_outflow_multiplier": float(best["eol_outflow_multiplier"]),
        "dmfa.fallback_calibration.new_scrap_fraction_of_demand": float(best["new_scrap_fraction_of_demand"]),
        "dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal": float(
            best["refined_export_cap_fraction"]
        ),
    }

    results.to_csv(out_dir / "calibration_metrics_all_candidates.csv", index=False)
    top.to_csv(out_dir / "calibration_metrics_top_candidates.csv", index=False)
    with open(out_dir / "best_candidate.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "scenario": args.scenario,
                "historic_window": {"start_year": hist_start, "end_year": hist_end},
                "run_window": {"start_year": run_start, "end_year": run_end},
                "objective": {
                    "type": "weighted_mape",
                    "weight_i_use": float(w_i_use),
                    "weight_stock": float(w_stock),
                },
                "best_candidate": best,
                "recommended_parameter_overrides": best_params,
            },
            f,
            sort_keys=False,
        )

    with open(out_dir / "calibration_assumptions.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "status": "TEMP",
                "notes": [
                    "Calibration objective uses weighted MAPE on historic i_use vs GAS observed and in_use_stock vs observed stock.",
                    "Search grid is coarse first-pass and intended for narrowing before final calibration.",
                ],
                "objective_weights": {
                    "i_use": float(w_i_use),
                    "stock": float(w_stock),
                },
                "grid": {
                    "utilization_target": util_grid,
                    "eol_outflow_multiplier": eol_mult_grid,
                    "refined_export_cap_fraction": ref_exp_grid,
                    "new_scrap_fraction_of_demand": new_scrap_grid,
                },
            },
            f,
            sort_keys=False,
        )

    print(f"Wrote calibration artifacts to: {out_dir}")
    print("Best candidate:")
    print(top.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
