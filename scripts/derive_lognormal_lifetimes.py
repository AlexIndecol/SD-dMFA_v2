"""Derive lognormal mu/sigma from mean/min/max using the MISO2 'range rule' approach.

This script reads:
- data/exogenous/lifetime_mean_years.csv
- data/exogenous/lifetime_min_years.csv
- data/exogenous/lifetime_max_years.csv

and writes:
- data/exogenous/lifetime_lognormal_mu.csv
- data/exogenous/lifetime_lognormal_sigma.csv

Method (standard lognormal moment relations + MISO2 range-rule for std):
1) Estimate standard deviation of a hypothetical sample:
   sd = (max - min) / 4        # 'range rule'
   var = sd^2
2) Convert to lognormal parameters:
   sigma^2 = ln(1 + var / mean^2)
   mu      = ln(mean) - 0.5*sigma^2

All inputs must be positive and satisfy max > min >= 0 and mean > 0.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import math

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exogenous"

IN_MEAN = DATA / "lifetime_mean_years.csv"
IN_MIN  = DATA / "lifetime_min_years.csv"
IN_MAX  = DATA / "lifetime_max_years.csv"
OUT_MU  = DATA / "lifetime_lognormal_mu.csv"
OUT_SG  = DATA / "lifetime_lognormal_sigma.csv"

def main():
    mean_df = pd.read_csv(IN_MEAN)
    min_df  = pd.read_csv(IN_MIN)
    max_df  = pd.read_csv(IN_MAX)

    key_cols = ["r","m","j"]
    df = mean_df.merge(min_df, on=key_cols, suffixes=("_mean","_min")).merge(max_df, on=key_cols)
    df = df.rename(columns={"value":"value_max"}).rename(columns={"value_mean":"mean","value_min":"min","value_max":"max"})

    mu_rows=[]
    sg_rows=[]

    for _, row in df.iterrows():
        r,m,j = row["r"], row["m"], row["j"]
        mean = row["mean"]
        mn   = row["min"]
        mx   = row["max"]

        if pd.isna(mean) or pd.isna(mn) or pd.isna(mx):
            mu=np.nan; sg=np.nan
        else:
            if mean <= 0 or mx <= mn or mn < 0:
                raise ValueError(f"Invalid lifetime inputs for {r}/{m}/{j}: mean={mean}, min={mn}, max={mx}")
            sd = (mx - mn)/4.0
            var = sd*sd
            sigma2 = math.log(1.0 + var/(mean*mean))
            sg = math.sqrt(sigma2)
            mu = math.log(mean) - 0.5*sigma2

        mu_rows.append({"r":r,"m":m,"j":j,"value":mu})
        sg_rows.append({"r":r,"m":m,"j":j,"value":sg})

    pd.DataFrame(mu_rows).to_csv(OUT_MU, index=False)
    pd.DataFrame(sg_rows).to_csv(OUT_SG, index=False)
    print(f"Wrote {OUT_MU} and {OUT_SG}")

if __name__ == "__main__":
    main()
