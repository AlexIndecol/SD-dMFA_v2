from __future__ import annotations
from pathlib import Path
import shutil
import datetime
import yaml

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def snapshot_configs(configs_dir: Path, out_dir: Path) -> None:
    dst = out_dir / "configs_snapshot"
    dst.mkdir(parents=True, exist_ok=True)
    for p in configs_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(configs_dir)
            (dst / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)

def write_metadata(out_dir: Path, metadata: dict) -> None:
    with open(out_dir / "run_metadata.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
