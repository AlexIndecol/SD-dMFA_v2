from __future__ import annotations
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


_DEFAULT_MISSING_POLICY = {
    "before_first_year": "hold_first",
    "after_last_year": "hold_last",
    "inside_gaps": "interpolate_linear",
    "require_full_coverage": False,
}

def list_exogenous_csvs(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])

def read_exogenous_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" not in df.columns:
        raise ValueError(f"{path.name} must include a 'value' column")
    # Keep the canonical `value` column while coercing blanks/non-numeric to NaN.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _resolve_missing_policy(data_or_paths: Any) -> dict[str, Any]:
    policy = dict(_DEFAULT_MISSING_POLICY)
    cfg_dir = getattr(data_or_paths, "configs", None)
    if cfg_dir is None:
        return policy

    cfg_path = Path(cfg_dir) / "data_sources.yml"
    if not cfg_path.exists():
        return policy

    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return policy

    cfg_policy = ((loaded.get("exogenous", {}) or {}).get("missing_policy", {}) or {})
    if not isinstance(cfg_policy, dict):
        return policy

    for k in _DEFAULT_MISSING_POLICY:
        if k in cfg_policy:
            policy[k] = cfg_policy[k]
    return policy


def _apply_missing_policy_one_group(group: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    if "t" not in group.columns:
        return group

    g = group.copy()
    g["value"] = pd.to_numeric(g["value"], errors="coerce")
    g["_t_num"] = pd.to_numeric(g["t"], errors="coerce")
    g = g.sort_values("_t_num", kind="mergesort")

    s = g["value"].copy()
    inside = str(policy.get("inside_gaps", "interpolate_linear")).strip().lower()
    before = str(policy.get("before_first_year", "hold_first")).strip().lower()
    after = str(policy.get("after_last_year", "hold_last")).strip().lower()

    if inside == "interpolate_linear":
        s = s.interpolate(method="linear", limit_area="inside")
    elif inside == "hold_last":
        s = s.ffill()
    elif inside == "hold_first":
        s = s.bfill()

    if before == "hold_first":
        s = s.bfill()
    if after == "hold_last":
        s = s.ffill()

    g["value"] = s
    g = g.drop(columns=["_t_num"])
    return g


def _apply_missing_policy(df: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    if "value" not in df.columns or "t" not in df.columns:
        return df

    out = df.copy()
    group_cols = [c for c in out.columns if c not in {"t", "value"}]
    if group_cols:
        parts = []
        for _, g in out.groupby(group_cols, dropna=False, sort=False):
            parts.append(_apply_missing_policy_one_group(g, policy))
        out = pd.concat(parts, ignore_index=False) if parts else out
    else:
        out = _apply_missing_policy_one_group(out, policy)

    out = out.sort_values([c for c in df.columns if c in out.columns], kind="mergesort")
    out = out[df.columns.tolist()]

    require_full = bool(policy.get("require_full_coverage", False))
    if require_full and pd.to_numeric(out["value"], errors="coerce").isna().any():
        raise ValueError("Missing-policy requested full coverage, but unresolved NaN values remain.")
    return out

def _resolve_data_dir(data_or_paths: Any) -> Path:
    if isinstance(data_or_paths, Path):
        return data_or_paths
    if hasattr(data_or_paths, "data_exogenous"):
        return Path(data_or_paths.data_exogenous)
    raise TypeError(f"Unsupported data source type: {type(data_or_paths)}")


def read_all_exogenous(data_or_paths: Any, *_args: Any, **_kwargs: Any) -> dict[str, pd.DataFrame]:
    data_dir = _resolve_data_dir(data_or_paths)
    policy = _resolve_missing_policy(data_or_paths)
    out = {}
    for p in list_exogenous_csvs(data_dir):
        out[p.stem] = _apply_missing_policy(read_exogenous_csv(p), policy)
    return out
