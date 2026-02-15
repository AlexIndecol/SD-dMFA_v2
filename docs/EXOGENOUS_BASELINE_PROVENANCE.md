# Baseline Exogenous Provenance (Historic)

This document records how required historic baseline exogenous series are generated for:
- `in_use_stock_observed_kt`
- `gas_to_use_observed_kt_per_yr`
- `od_preference_weight_0_1`
- `capacity_stage_observed_kt_per_yr`
- `lifetime_lognormal_mu`
- `lifetime_lognormal_sigma`

Primary implementation:
- `scripts/data/fill_baseline_required_exogenous.py`

## Scope and windows
- Historic window used for required baseline population: `1870..2019`.
- Files keep template rows outside this window blank where not required for historic calibration.
- Model units for mass flows/stocks: `kt` and `kt/yr`.

## Unit normalization
Raw BGS and OWID mass values are ingested as tonnes and converted to `kt`:
- `kt = tonnes / 1000`

Applied in:
- `load_bgs_production()`
- `load_bgs_trade()`
- `load_owid_world_mine()`

## Source datasets
- BGS production/trade:  
  `/Users/alexcolloricchio/Desktop/CRMs/Data/bgs/*.csv`
- OWID mine production:  
  `/Users/alexcolloricchio/Desktop/CRMs/Data/ourworldindata/global-mine-production-minerals.zip`
- MISO end-use shares:  
  `/Users/alexcolloricchio/Desktop/CRMs/Data/MISO2_v1_lifetimes_wasteRates_recycling/MISO2_v1_2_EoL_endUse.xlsx`
- MISO lifetime means:  
  `/Users/alexcolloricchio/Desktop/CRMs/Data/MISO2_v1_lifetimes_wasteRates_recycling/MISO2_Lifetimes_v1.xlsx`
- MISO lifetime deviations (sigma proxy):  
  `/Users/alexcolloricchio/Desktop/CRMs/Data/MISO2_v1_lifetimes_wasteRates_recycling/MISO2_Lifetimes_deviation_v1.xlsx`
- BACI trade data:  
  `/Users/alexcolloricchio/Desktop/CRMs/Data/baci/BACI_HS92_V202601.zip`  
  `/Users/alexcolloricchio/Desktop/CRMs/Data/baci/BACI_HS22_V202601.zip`
- HS concordance (official UNSD):  
  `https://unstats.un.org/unsd/classifications/Econ/tables/HS-SITC-BEC%20Correlations_2022.xlsx`

## Variable derivations

### gas_to_use_observed_kt_per_yr
1. Build `refining_primary` proxy from BGS production tables using rows tagged by smelter/refinery/slab labels.
2. Build refined-metal imports and exports from BGS trade files.
3. Compute total GAS by `(t,r,m)`:
   - `gas_total = max(0, refining_primary + imports_refined - exports_refined)`
4. Split `gas_total` to `(t,r,m,j)` using MISO-based end-use shares.

Output:
- `data/exogenous/gas_to_use_observed_kt_per_yr.csv`

### in_use_stock_observed_kt
1. Use `gas_to_use` as inflow `I_t` by `(r,m,j)`.
2. Use MISO lifetime mean `L` by `(r,m,j)`.
3. Apply stock recursion:
   - initialization: `S_0 = I_0 * L`
   - update: `S_t = S_{t-1} + I_t - S_{t-1}/L`
   - clip at zero if numerical negatives occur.

Output:
- `data/exogenous/in_use_stock_observed_kt.csv`

### lifetime_lognormal_sigma
1. Aggregate MISO lifetime deviation dataset to model `(r,m,j)` using detailed-sector weights.
2. Bound sigma to stable range:
   - `0.05 <= sigma <= 1.5`

Output:
- `data/exogenous/lifetime_lognormal_sigma.csv`

### lifetime_lognormal_mu
1. Use aggregated lifetime mean and sigma by `(r,m,j)`.
2. Convert to lognormal mu:
   - `mu = ln(mean) - 0.5 * sigma^2`

Output:
- `data/exogenous/lifetime_lognormal_mu.csv`

### capacity_stage_observed_kt_per_yr
Derived stage capacities from mine/refining/GAS proxies plus collection and EoL recycling rates.
- Primary extraction and beneficiation from mine production proxy.
- Refining primary from refining proxy.
- Fabrication/use from GAS proxy.
- Collection/recycling/residue/environment from configured baseline rates.

Output:
- `data/exogenous/capacity_stage_observed_kt_per_yr.csv`

### od_preference_weight_0_1
1. Build bilateral BACI flows for target commodity baskets (Zn/Ni/Sn, concentrate/refined/scrap).
2. Link HS92 and HS22 code systems via official UNSD concordance.
3. Stitch series:
   - HS92 for years `< 2022`
   - HS22 for years `>= 2022`
4. Pre-BACI period fallback:
   - gravity-style OD from BGS import/export marginals.
5. Normalize OD row-wise:
   - for each `(t,m,c,o)`, `sum_d w = 1`.

Output:
- `data/exogenous/od_preference_weight_0_1.csv`

Supporting artifacts:
- `data/reference/baci_hs22_to_hs92_concordance_used.csv`
- `data/reference/baci_hs92_vs_hs22_overlap_diagnostics.csv`

## Cross-checks and validation
Executed in ETL run:
- `max |sum_j gas_j - gas_total|` close to numerical zero.
- `max |sum_d od_weight - 1|` close to numerical zero.
- BGS vs OWID mine overlap relative error summary.

Repository validator:
- `python scripts/validate_exogenous_inputs.py`

## Assumptions and decisions
Confirmed choices (see `configs/assumptions.yml`):
- OD weights use BACI trade value (`v`), not quantity (`q`).
- Tin refined basket includes HS `800300`.
- HS stitch rule uses HS92 `<2022` and HS22 `>=2022` with no chain-link scaling.
