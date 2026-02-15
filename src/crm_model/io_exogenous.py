from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

def list_exogenous_csvs(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])

def read_exogenous_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" not in df.columns:
        raise ValueError(f"{path.name} must include a 'value' column")
    # Keep the canonical `value` column while coercing blanks/non-numeric to NaN.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def _resolve_data_dir(data_or_paths: Any) -> Path:
    if isinstance(data_or_paths, Path):
        return data_or_paths
    if hasattr(data_or_paths, "data_exogenous"):
        return Path(data_or_paths.data_exogenous)
    raise TypeError(f"Unsupported data source type: {type(data_or_paths)}")

def read_all_exogenous(data_or_paths: Any, *_args: Any, **_kwargs: Any) -> dict[str, pd.DataFrame]:
    data_dir = _resolve_data_dir(data_or_paths)
    out = {}
    for p in list_exogenous_csvs(data_dir):
        out[p.stem] = read_exogenous_csv(p)
    return out
