from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Paths:
    root: Path
    configs: Path
    data_exogenous: Path
    outputs: Path

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_paths(root: Path) -> Paths:
    return Paths(
        root=root,
        configs=root / "configs",
        data_exogenous=root / "data" / "exogenous",
        outputs=root / "outputs",
    )
