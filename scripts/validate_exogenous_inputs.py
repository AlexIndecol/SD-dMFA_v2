"""Quick input validation (headers, codes, share constraints).

Usage:
  python scripts/validate_exogenous_inputs.py

This validates CSVs against configs/dimensions.yml and checks share constraints
for rows with numeric values.
"""

from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIM = ROOT / "configs" / "dimensions.yml"
DATA = ROOT / "data" / "exogenous"

REQUIRED = [
  "in_use_stock_observed_kt.csv",
  "gas_to_use_observed_kt_per_yr.csv",
  "use_share_j_frac.csv",
  "capacity_stage_observed_kt_per_yr.csv",
  "od_preference_weight_0_1.csv",
  "trade_factor_i_ge_1.csv",
  "wgi_norm_i_0_1.csv",
  "lifetime_mean_years.csv",
  "lifetime_lognormal_mu.csv",
  "lifetime_lognormal_sigma.csv",
  "reserves_kt.csv",
  "resources_kt.csv",
  "ore_mined_kt_per_yr.csv",
  "nonsubstitutability_j_0_1.csv",
]

OPTIONAL = [
  "end_use_detail_share_frac.csv",
  "lifetime_min_years.csv",
  "lifetime_max_years.csv",
  "byproduct_primary_production_kt_per_yr.csv",
]

def load_yaml(p):
  with open(p, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)

def numeric_only(df, col="value"):
  s = pd.to_numeric(df[col], errors="coerce")
  out = df[s.notna()].copy()
  out[col] = s[s.notna()].astype(float).values
  return out

def check_exists():
  for fn in REQUIRED:
    if not (DATA/fn).exists():
      raise FileNotFoundError(DATA/fn)

def main():
  cfg = load_yaml(DIM)
  regions = set(cfg["dimensions"]["regions"]["values"])
  materials = set(cfg["dimensions"]["materials"]["values"])
  end_uses = set(cfg["dimensions"]["end_uses"]["values"])
  end_use_detail = set(cfg["dimensions"].get("end_use_detail",{}).get("values", []))
  stages = set(cfg["dimensions"]["stages"]["values"])
  commodities = set(cfg["dimensions"]["commodities"]["values"])
  suppliers = set(cfg["dimensions"]["suppliers"]["values"])
  origins = set(cfg["dimensions"]["origins"]["values"])
  destinations = set(cfg["dimensions"]["destinations"]["values"])

  check_exists()

  # use_share sums over j for each (t,m)
  df = pd.read_csv(DATA/"use_share_j_frac.csv")
  dfn = numeric_only(df)
  if not dfn.empty:
    s = dfn.groupby(["t","m"])["value"].sum()
    bad = s[(s < 0.999) | (s > 1.001)]
    if len(bad) > 0:
      raise ValueError(f"use_share_j_frac does not sum to 1 for {len(bad)} keys (first few):\n{bad.head()}")

  # end_use_detail shares sums over jd for each (t,r,m,j)
  if (DATA/"end_use_detail_share_frac.csv").exists():
    df = pd.read_csv(DATA/"end_use_detail_share_frac.csv")
    dfn = numeric_only(df)
    if not dfn.empty:
      s = dfn.groupby(["t","r","m","j"])["value"].sum()
      bad = s[(s < 0.999) | (s > 1.001)]
      if len(bad) > 0:
        raise ValueError(f"end_use_detail_share_frac does not sum to 1 for {len(bad)} keys (first few):\n{bad.head()}")

  # basic code checks (only for present columns)
  def check_codes(df, col, allowed, name):
    if col in df.columns:
      vals = set(df[col].dropna().astype(str).unique())
      unk = {v for v in vals if v not in allowed}
      if unk:
        raise ValueError(f"{name}: unknown codes in '{col}': {sorted(unk)[:10]}")

  # spot-check a few key files
  df = pd.read_csv(DATA/"in_use_stock_observed_kt.csv")
  check_codes(df,"r",regions,"in_use_stock_observed_kt")
  check_codes(df,"m",materials,"in_use_stock_observed_kt")
  check_codes(df,"j",end_uses,"in_use_stock_observed_kt")

  df = pd.read_csv(DATA/"capacity_stage_observed_kt_per_yr.csv")
  check_codes(df,"r",regions,"capacity_stage_observed_kt_per_yr")
  check_codes(df,"m",materials,"capacity_stage_observed_kt_per_yr")
  check_codes(df,"p",stages,"capacity_stage_observed_kt_per_yr")

  df = pd.read_csv(DATA/"od_preference_weight_0_1.csv")
  check_codes(df,"m",materials,"od_preference_weight_0_1")
  check_codes(df,"c",commodities,"od_preference_weight_0_1")
  check_codes(df,"o",origins,"od_preference_weight_0_1")
  check_codes(df,"d",destinations,"od_preference_weight_0_1")

  print("Validation OK (blank templates allowed; constraints checked only for numeric rows).")

if __name__ == "__main__":
  main()
