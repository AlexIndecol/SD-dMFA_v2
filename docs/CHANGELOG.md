# Changelog

## v4.9
- Filled required baseline historic exogenous inputs using CRM source data and internal cross-checks:
  - `data/exogenous/in_use_stock_observed_kt.csv`
  - `data/exogenous/gas_to_use_observed_kt_per_yr.csv`
  - `data/exogenous/od_preference_weight_0_1.csv`
  - `data/exogenous/capacity_stage_observed_kt_per_yr.csv`
  - `data/exogenous/lifetime_lognormal_mu.csv`
  - `data/exogenous/lifetime_lognormal_sigma.csv`
- Implemented BACI trade ingestion linked through official UNSD concordance (`HS92` <-> `HS22`) in:
  - `scripts/data/fill_baseline_required_exogenous.py`
- Added BACI trade concordance and overlap diagnostics artifacts:
  - `data/reference/baci_hs22_to_hs92_concordance_used.csv`
  - `data/reference/baci_hs92_vs_hs22_overlap_diagnostics.csv`
- Fixed data generation bugs in the baseline fill pipeline:
  - corrected in-use stock inflow handling (prevented zero-stock artifact),
  - corrected merge-suffix issues in time-extension helpers,
  - corrected multi-material duplication in backcast extension.
- Corrected mass-unit normalization to `kt` in baseline ETL:
  - BGS production/trade and OWID mine-production source values are ingested as tonnes and converted to `kt` before writing exogenous CSVs.
  - Regenerated baseline required historic exogenous files after conversion.
- Added report-support cross-check:
  - Tin supplementary dataset (`jiec13459-sup-0002-suppmat.xlsx`) 2017 reference values are now consistent in magnitude with generated Sn stage capacities after unit conversion.
- Added explicit provenance documentation for required baseline historic exogenous series:
  - `docs/EXOGENOUS_BASELINE_PROVENANCE.md`
  - linked from variable docs in `data/exogenous/*.md`.
- Added workflow visualization docs:
  - `docs/WORKFLOW_VISUALS.md` (embedded Mermaid diagrams for coupled model and baseline exogenous ETL),
  - `docs/diagrams/exogenous_baseline_workflow.mmd`.
- Added a detailed full-chain architecture diagram for dMFA + SD coupling + OD trade:
  - `docs/diagrams/coupled_full_chain_scope.mmd`,
  - embedded in `docs/WORKFLOW_VISUALS.md`.
- Made fallback dMFA calibration knobs explicit in config (default behavior unchanged):
  - `configs/parameters/parameters_dmfa.yml` now includes `fallback_calibration` with
    `new_scrap_fraction_of_demand`, `eol_outflow_multiplier`, and commodity export-cap fractions.
- Updated fallback dMFA implementation to read calibration knobs from config:
  - `src/crm_model/dmfa_model.py`.
- Added reproducible baseline calibration sweep script:
  - `scripts/calibration/run_baseline_calibration_loop.py`.
- Executed first coarse calibration sweep and exported ranked candidates:
  - `outputs/calibration/baseline/20260216_091918/*`.
- Recorded user calibration choices (`1.A`, `2.B`, `3.B`) and executed constrained second-pass calibration (equal weights, `utilization_target <= 1.0`) without applying overrides yet:
  - `outputs/calibration/baseline/20260216_093512/*` (detailed constrained sweep),
  - `outputs/calibration/baseline/20260216_094654/*` (constrained quick-check sweep).
- Executed an additional focused constrained sweep after user selected non-adoption (`Option B`) to refine `new_scrap_fraction_of_demand` and `refined_export_cap_fraction`:
  - `outputs/calibration/baseline/20260216_100712/*`,
  - focused best objective improved to `0.220885` under `utilization_target=1.0`.
- Adopted focused constrained best candidate into baseline parameter configs and reran full baseline:
  - `configs/parameters/parameters_sd.yml`: `utilization_target = 1.0`
  - `configs/parameters/parameters_dmfa.yml`: `new_scrap_fraction_of_demand = 0.07`, `refined_export_cap_fraction = 0.19`, `eol_outflow_multiplier = 1.0`
  - run output: `outputs/baseline/20260216_101506/*`
- Added preconfigured inspection notebooks in-repo:
  - `notebooks/00_Quickstart.ipynb` through `notebooks/06_Indicators_Viewer.ipynb`,
  - `notebooks/README.md`.
- Patched notebook root path handling so notebooks work from either repo root or `notebooks/`.
- Closed BACI trade assumption confirmations (user selected 1.A, 2.A, 3.A):
  - keep value-based OD weights (`v`),
  - include `800300` in tin `refined_metal`,
  - keep HS92<2022 and HS22>=2022 without chain-link scaling.

## v4.8
- Implemented a first executable annual coupled engine across SD, dMFA, and OD trade allocation:
  - `src/crm_model/coupling.py`: concrete `run_coupled_year` orchestration with shared stabilizer smoothing and state carry-over.
  - `src/crm_model/sd_model.py`: deterministic fallback SD step (demand, capacities, rates, lifetime multiplier, price feedback).
  - `src/crm_model/dmfa_model.py`: deterministic fallback dMFA step (I_use, EoL outflows, commodity stocks, scrap split, production, export caps).
- Fixed exogenous ingestion interface mismatch and numeric coercion:
  - `src/crm_model/io_exogenous.py` now supports the `run.py` call signature and coerces `value` to numeric (blank -> NaN).
- Reworked run pipeline:
  - `src/crm_model/run.py` now loads merged parameters/scenarios, runs full annual coupling in non-dry mode, computes indicators, and exports required run artifacts.
  - Added stop-the-run enforcement for missing/empty required historic exogenous inputs unless TEMP-approved in `configs/assumptions.yml`.
  - Added per-run `assumptions_used.yml` and `run_note.md`.
- Improved CLI robustness:
  - `src/crm_model/cli.py` now falls back to `argparse` when `typer` is unavailable.
- Added governance docs required by `AGENTS.md`:
  - `docs/TASK_BOARD.md`
  - `docs/DECISION_LOG.md`
  - `docs/RISKS.md`
  - `docs/ASSUMPTIONS.md`

## v4.7
- Confirmed modelling decisions in configs/assumptions.yml:
  - OD matrices are **weights** (relative preferences), not shares.
  - Trade commodities are **concentrate**, **scrap**, **refined_metal**.
  - Reuse/reman is represented via **lifetime extension** (lognormal mu shift by ln(k)).
  - Demand-based indicators use **realized demand = I_use**.
- Added SD/dMFA/trade/coupling **module interfaces** in `src/crm_model/` (config-driven; equations remain to be implemented).
- Implemented a simple OD **weights allocator** (`allocate_od_weights`) supporting commodity-specific export caps.
- Added accounting scaffolding for **commodity buffers**, **balancing items**, and **scrap split/delay** (variables + coupling contract).
- Extended indicator config with v4.7 add-ons: **UR**, demand-based DIR (realized), balancing diagnostics, and buffer coverage.
- Added new Jupyter cookbooks (`notebooks/07–09`) and updated Mermaid workflow to reflect buffers/balancing.


## v4.6
- Added scripts/validate_consistency.py (indicator requires + scenario registry checks).
- Added docs/CONSISTENCY_SCAN_v4_6.md and auto-added placeholder variables to ensure internal consistency.


## v4.5
- Parsed INDICATORS.md into configs/indicators.yml (45 indicators/metrics + inferred dims/requires).
- Updated docs/INDICATORS_SPEC.md to reflect auto-parsing and conventions.


## v4.4
- Added AGENTS.md (compact governance rules).
- Added configs/assumptions.yml and configs/reporting.yml.
- Added scenario blueprints for R-strategies and disruptions (configs/scenarios/*).
- Added combined policy package and compound shock scenario blueprints.
