# Workflow Visuals

This page gives a quick visual of:
- the coupled annual model loop,
- the full-chain dMFA scope and SD coupling interactions, and
- the baseline historic exogenous data build pipeline.

## 1) Coupled model annual workflow

```mermaid
graph LR
  SD[SD]
  DMFA[dMFA]
  TRADE[OD Trade]
  IND[Indicators]

  SD -->|sd_to_dmfa| DMFA
  DMFA -->|dmfa_to_sd| SD
  TRADE -->|trade_outputs| SD

  SD --> IND
  DMFA --> IND
  TRADE --> IND
```

Reference file:
- `docs/diagrams/workflow.mmd` (auto-generated from `configs/coupling.yml` via `scripts/generate_mermaid_workflow.py`)

## 1a) Full-chain dMFA scope + SD coupling (detailed)

```mermaid
flowchart LR
  subgraph EXO_FILE[File Exogenous Inputs data/exogenous]
    F_GAS[gas_to_use_observed_kt_per_yr]
    F_CAPOBS[capacity_stage_observed_kt_per_yr]
    F_USESH[use_share_j_frac optional]
    F_LIFEMU[lifetime_lognormal_mu]
    F_LIFESIG[lifetime_lognormal_sigma]
    F_LIFEMEAN[lifetime_mean_years optional]
    F_ORE[ore_mined_kt_per_yr optional]
    F_ODW[od_preference_weight_0_1]
    F_TF[trade_factor_i_ge_1]
    F_RES[resources_kt indicator input]
    F_RESV[reserves_kt indicator input]
  end

  subgraph EXO_RUNTIME[Runtime Exchange Variables generated during run]
    X_DEMAND[demand_kt_per_yr]
    X_COLL[collection_rate_0_1]
    X_REC[recovery_rate_0_1]
    X_YIELD[recycling_yield_0_1]
    X_LIFE[lifetime_multiplier_ge_1]
    X_CAPRAW[capacity_stage_raw_kt_per_yr]
    X_PRICE[price_index_rel optional]
  end

  subgraph SD[SD Model annual dynamics]
    SD_DEM[Demand and policy state]
    SD_SCAR[Scarcity and price perception]
    SD_LEV[Levers collection recovery capacity lifetime]
  end

  subgraph DMFA[dMFA full chain]
    P1[primary_extraction]
    P2[beneficiation_concentration]
    P3[refining_primary]
    P4[fabrication_and_manufacturing]
    P5[use_phase and in_use_stock]
    P6[collection]
    P7[sorting_preprocessing]
    P8[recycling_refining_secondary]
    P9[residue_treatment_disposal]
    P10[environment]
  end

  subgraph STAB[Coupling stabilizer]
    STAB_SEC[s_sec_max_to_sd_kt_per_yr]
    STAB_CAP[capacity_stage_to_sd_kt_per_yr]
    STAB_EXP[od_export_cap_kt_per_yr]
    STAB_PRICE[price_to_sd_smoothed_rel optional]
  end

  subgraph TRADE[OD Trade allocator]
    TR_IN[OD weights frictions export caps]
    TR_ALLOC[Allocate OD flows by commodity]
    TR_OUT[imports exports share_i_frac]
  end

  subgraph IND[Indicators]
    IND1[supply concentration]
    IND2[circularity and recycling]
    IND3[demand fulfilment]
    IND4[buffers losses and stocks]
  end

  P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7
  P7 --> P8 --> P4
  P7 --> P9 --> P10

  F_GAS --> SD_DEM
  F_CAPOBS --> SD_LEV
  F_USESH --> SD_DEM
  F_ODW --> TR_IN
  F_TF --> TR_IN
  F_LIFEMU --> P5
  F_LIFESIG --> P5
  F_LIFEMEAN --> P5
  F_ORE --> P1
  F_RES --> IND1
  F_RESV --> IND1

  SD_DEM --> SD_SCAR
  SD_SCAR --> SD_LEV
  SD_LEV --> SD_DEM

  SD_DEM --> X_DEMAND
  SD_LEV --> X_COLL
  SD_LEV --> X_REC
  SD_LEV --> X_YIELD
  SD_LEV --> X_LIFE
  SD_LEV --> X_CAPRAW
  SD_SCAR --> X_PRICE

  X_DEMAND -->|sd_to_dmfa| P4
  X_COLL --> P6
  X_REC --> P7
  X_YIELD --> P8
  X_LIFE --> P5
  X_CAPRAW --> P1
  X_PRICE --> P4

  P7 -->|s_sec_max_raw_kt_per_yr| STAB_SEC
  P1 -->|capacity signals| STAB_CAP
  P2 -->|capacity signals| STAB_CAP
  P3 -->|capacity signals| STAB_CAP
  X_PRICE -->|price smoothing input| STAB_PRICE

  STAB_SEC --> SD_SCAR
  STAB_CAP --> SD_LEV
  STAB_CAP --> STAB_EXP
  STAB_EXP --> TR_IN
  STAB_PRICE --> SD_SCAR

  P2 -->|export concentrate availability| TR_IN
  P3 -->|export refined availability| TR_IN
  P7 -->|export scrap availability| TR_IN
  TR_IN --> TR_ALLOC --> TR_OUT

  TR_OUT -->|imports concentrate| P3
  TR_OUT -->|imports refined_metal| P4
  TR_OUT -->|imports scrap| P8
  TR_OUT -->|trade_outputs to SD| SD_SCAR

  P10 -->|total_losses_kt_per_yr| SD_SCAR
  P5 -->|in_use_stock_kt| SD_SCAR
  P6 -->|eol_outflow_kt_per_yr| SD_SCAR
  P4 -->|i_use_kt_per_yr unmet_demand_kt_per_yr| SD_DEM
  P3 -->|stock_refined_metal_kt| SD_SCAR
  P7 -->|stock_scrap_kt| SD_SCAR
  P2 -->|stock_concentrate_kt| SD_SCAR

  SD_DEM --> IND3
  SD_SCAR --> IND4
  P5 --> IND2
  P7 --> IND2
  TR_OUT --> IND1
  P10 --> IND4
```

Reference file:
- `docs/diagrams/coupled_full_chain_scope.mmd`

## 2) Baseline exogenous ETL workflow (historic)

```mermaid
graph TD
  subgraph Sources
    BGSProd[BGS production csv]
    BGSTrade[BGS imports exports csv]
    OWID[OWID mine production]
    MISOShare[MISO end use shares]
    MISOLife[MISO lifetime data]
    BACI92[BACI HS92]
    BACI22[BACI HS22]
    UNSD[UNSD HS concordance]
  end

  subgraph Transform
    Units[Convert tonnes to kt]
    MineRef[Mine and refining proxies]
    TradeMR[Refined imports and exports]
    GAS[GAS total]
    SplitJ[Split GAS to end uses]
    StockDyn[Stock recursion]
    LifeAgg[Aggregate lifetime stats]
    MuCalc[Compute lognormal mu]
    ODLink[Link HS92 and HS22]
    ODFallback[Pre BACI fallback]
    ODNorm[Normalize OD weights]
    CapMap[Map stage capacities]
    Checks[Internal checks + validator]
  end

  subgraph Outputs
    GASCSV[gas_to_use_observed_kt_per_yr.csv]
    STOCKCSV[in_use_stock_observed_kt.csv]
    ODC[od_preference_weight_0_1.csv]
    CAPCSV[capacity_stage_observed_kt_per_yr.csv]
    MUCSV[lifetime_lognormal_mu.csv]
    SIGCSV[lifetime_lognormal_sigma.csv]
  end

  BGSProd --> Units
  BGSTrade --> Units
  Units --> MineRef
  Units --> TradeMR
  OWID --> Units
  MISOShare --> SplitJ
  MISOLife --> LifeAgg
  LifeAgg --> MuCalc
  BACI92 --> ODLink
  BACI22 --> ODLink
  UNSD --> ODLink

  MineRef --> GAS
  TradeMR --> GAS
  GAS --> SplitJ
  SplitJ --> GASCSV
  SplitJ --> StockDyn
  StockDyn --> STOCKCSV

  LifeAgg --> SIGCSV
  MuCalc --> MUCSV

  ODLink --> ODNorm --> ODC
  ODFallback --> ODNorm
  BGSTrade --> ODFallback

  MineRef --> CapMap --> CAPCSV
  GAS --> CapMap

  GASCSV --> Checks
  STOCKCSV --> Checks
  ODC --> Checks
  CAPCSV --> Checks
  MUCSV --> Checks
  SIGCSV --> Checks
```

Reference:
- `docs/EXOGENOUS_BASELINE_PROVENANCE.md`

If your Mermaid extension still fails, open the standalone files directly:
- `docs/diagrams/workflow.mmd`
- `docs/diagrams/coupled_full_chain_scope.mmd`
- `docs/diagrams/exogenous_baseline_workflow.mmd`
