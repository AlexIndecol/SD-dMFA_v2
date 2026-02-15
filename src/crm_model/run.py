from __future__ import annotations
from pathlib import Path
import pandas as pd

from .config import get_paths, load_yaml
from .io_exogenous import read_all_exogenous
from .snapshot import timestamp, snapshot_configs, write_metadata
from .indicators import compute_indicators

def run_project(root: Path, scenario_id: str, dry_run: bool = True) -> Path:
    """Run a configured project.
    - dry_run=True: ingest exogenous data, snapshot configs, compute any indicators possible from available series.
    - dry_run=False: reserved for coupled SD+dMFA+trade execution (v4.7 provides interfaces but not equations).
    """
    paths = get_paths(root)

    # Load core configs
    time_cfg = load_yaml(paths.configs / "time.yml")
    coupling_cfg = load_yaml(paths.configs / "coupling.yml")
    ind_cfg = load_yaml(paths.configs / "indicators.yml")

    # Resolve scenario
    sc_index = load_yaml(paths.configs / "scenarios" / "index.yml")
    scenario_file = sc_index.get("scenarios", {}).get(scenario_id)
    scenario_cfg = load_yaml(root / scenario_file) if scenario_file else {"id": scenario_id, "inherits": None}

    # Output folder
    out_dir = paths.outputs / scenario_id / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot configs (reproducibility)
    snapshot_configs(paths.configs, out_dir)
    write_metadata(out_dir, {
        "scenario_id": scenario_id,
        "dry_run": dry_run,
        "time": time_cfg.get("time", {}),
        "periods": time_cfg.get("periods", {}),
        "note": "v4.7: dry-run computes a small subset of indicators from available series; full coupled equations not yet implemented."
    })

    # Ingest exogenous series
    exo = read_all_exogenous(paths, time_cfg, scenario_cfg)

    # Compute indicators (subset implementation; no silent drops)
    indicators = compute_indicators(ind_cfg, exo)

    ind_dir = out_dir / "indicators"
    ind_dir.mkdir(parents=True, exist_ok=True)
    for name, df in indicators.items():
        df.to_csv(ind_dir / f"{name}.csv", index=False)

    # Write series snapshot (optional)
    snap_dir = out_dir / "series_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for k, df in exo.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(snap_dir / f"{k}.csv", index=False)

    return out_dir
