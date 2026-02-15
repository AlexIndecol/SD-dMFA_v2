from __future__ import annotations
from pathlib import Path
import pandas as pd

def list_exogenous_csvs(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])

def read_exogenous_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" not in df.columns:
        raise ValueError(f"{path.name} must include a 'value' column")
    return df

def read_all_exogenous(data_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for p in list_exogenous_csvs(data_dir):
        out[p.stem] = read_exogenous_csv(p)
    return out
