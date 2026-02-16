# Changelog

## v5.3
- Migrated dMFA native runtime to explicit full flodym `MFASystem` process-network execution:
  - `src/crm_model/dmfa_model.py` now builds a stage process network (`sysenv` + configured stages) with named flows across extraction, beneficiation, refining, fabrication/use, collection/sorting/recycling, residue treatment, environment, and trade-boundary interfaces.
  - Runtime now computes and exports `process_flow_kt_per_yr` (long-form process-edge flow series) for diagnostics and inspection.
- Coupling outputs now include process-network flow series:
  - `src/crm_model/coupling.py`: `process_flow_kt_per_yr`.
- Code cleanup/refactor (no behavior change):
  - `src/crm_model/sd_model.py`: reduced repeated backend checks and vectorized baseline-demand/elasticity assignments for readability.
  - `src/crm_model/dmfa_model.py`: centralized required stage/commodity constants and simplified process-flow definition construction.
- Verification reruns after MFASystem migration:
  - baseline: `outputs/baseline/20260216_213718`
  - scenario: `outputs/r_collection_push/20260216_213814`
  - both report `runtime_modules.*.execution_mode = native_required` and `use_native = true`.

## v5.2
- Final cleanup after fallback-execution purge:
  - `src/crm_model/dmfa_model.py`: removed dead lifetime-extension helper from runtime module and centralized calibration-block resolution (`calibration` first, legacy `fallback_calibration` still accepted).
  - `src/crm_model/coupling.py`: removed no-op export-cap rename and normalized stock-series slicing for readability.
  - `configs/parameters/parameters_dmfa.yml`: renamed active parameter block from `fallback_calibration` to `calibration`.
  - `scripts/calibration/run_baseline_calibration_loop.py`: switched calibration writes/overrides to `dmfa.calibration.*`.
- Verification reruns after cleanup:
  - baseline: `outputs/baseline/20260216_210400`
  - scenario: `outputs/r_collection_push/20260216_210435`
  - both report `runtime_modules.*.execution_mode = native_required` and `use_native = true`.

## v5.1
- Purged fallback execution mode from runtime code paths:
  - `src/crm_model/sd_model.py` now accepts only `execution_mode: native_required`.
  - `src/crm_model/dmfa_model.py` now accepts only `execution_mode: native_required`.
- Cleaned model implementation by removing legacy mixed fallback/native dMFA path and keeping a single native flodym-array execution path.
- Updated runtime metadata note in `src/crm_model/run.py` to explicitly state fallback execution removal.
- Verification reruns after cleanup:
  - baseline: `outputs/baseline/20260216_205554`
  - scenario: `outputs/r_collection_push/20260216_205632`
  - both report `runtime_modules.*.execution_mode = native_required` and `use_native = true`.

## v5.0
- Completed full native runtime migration for coupled annual runs:
  - `src/crm_model/sd_model.py`: native BPTK-Py execution is now required for SD run steps; fallback execution is disabled.
  - `src/crm_model/dmfa_model.py`: added full flodym-array native computation path for dMFA run steps; runtime fallback execution is disabled.
- Switched coupling defaults to strict native execution:
  - `configs/coupling.yml`: `modules.sd.execution_mode = native_required`
  - `configs/coupling.yml`: `modules.dmfa.execution_mode = native_required`
- Updated run metadata note/versioning for native-first runtime:
  - `src/crm_model/run.py`: run note now `v5.0` and dry-run defaults reflect `native_required`.
- Verification reruns:
  - baseline: `outputs/baseline/20260216_204820`
  - scenario: `outputs/r_collection_push/20260216_204635`
  - both report `runtime_modules.*.execution_mode = native_required` and `use_native = true`.

## v4.10
- Activated native framework dispatch wiring for coupled runs:
  - `src/crm_model/sd_model.py` now uses execution modes (`native_required`, `native_if_available`, `fallback_only`) and dispatches through BPTK-Py when available.
  - `src/crm_model/dmfa_model.py` now uses the same execution-mode contract and dispatches through flodym-backed array operations when available.
- Added explicit native-dispatch controls in `configs/coupling.yml`:
  - `modules.sd.execution_mode`
  - `modules.dmfa.execution_mode`
- Extended run artifacts to expose actual runtime engine use:
  - `src/crm_model/run.py` now writes `runtime_modules` in `run_metadata.yml` with `engine_requested`, `execution_mode`, `native_available`, and `use_native`.
  - Updated run note/metadata wording from fallback-only to native-dispatch aware.
- Verification reruns:
  - baseline: `outputs/baseline/20260216_203603`
  - scenario: `outputs/r_collection_push/20260216_203642`
  - both report `use_native: true` for SD and dMFA in `run_metadata.yml`.

## v4.9
- Closed headline computation gaps for updated reporting set:
  - `apparent_consumption_kt_per_yr` now computed in `src/crm_model/indicators.py` (prefers direct coupled accounting series, with formula fallback).
  - `stockpile_kt` and `stockpile_cover_years` now computed in `src/crm_model/indicators.py`.
  - Added `domestic_production_kt_per_yr` and `stock_change_kt_per_yr` to coupled outputs in `src/crm_model/coupling.py`.
- Added TEMP stockpile proxy governance:
  - `configs/assumptions.yml`: `stockpile_proxy_from_refined_buffer_temp`.
  - `docs/ASSUMPTIONS.md` updated accordingly.
- Verification rerun (latest scenario requested):
  - `outputs/r_collection_push/20260216_192200`
  - all current headlines in `configs/reporting.yml` are now numeric in this run.
- Fixed exogenous missing-policy enforcement in loader:
  - `src/crm_model/io_exogenous.py` now applies `configs/data_sources.yml -> exogenous.missing_policy`
    (`before_first_year`, `after_last_year`, `inside_gaps`) when reading exogenous CSVs.
- Fixed SD demand fallback gating:
  - `src/crm_model/sd_model.py` now uses observed GAS only when the year slice contains numeric values,
    preventing all-NaN GAS rows from collapsing to zero demand via grouped sums.
- Verification reruns after demand-path fix:
  - baseline: `outputs/baseline/20260216_153435`
  - scenario sweep (latest per scenario around `15:34:59..15:37:39`, 2026-02-16)
  - comparison rebuilt: `outputs/comparisons/20260216_153822/*`
- Resulting behavior change:
  - reporting-period `demand_kt_per_yr` is no longer forced to zero by NaN GAS rows;
    scenario headline deltas are materially non-zero for several scenarios.
- Implemented user-selected scenario calibration option `1.B` (TEMP conservative autofill for unresolved control magnitudes):
  - added `configs/scenario_autofill.yml`,
  - loaded/passed autofill config through `src/crm_model/run.py` and `src/crm_model/coupling.py`,
  - applied autofill at control-resolution and runtime application level in `src/crm_model/scenario_controls.py`.
- Extended run artifacts for TEMP autofill transparency:
  - `run_metadata.yml` now includes `scenario_controls_autofill_enabled` and `scenario_controls_autofill_used`,
  - `run_note.md` now includes autofill status in scenario-control summary.
- Fixed control application bounds so `_0_1` variables can increase via multipliers while still clipping final applied values to `[0,1]`.
- Verification outputs for `1.B` implementation:
  - `outputs/baseline/20260216_140102`
  - `outputs/r_collection_push/20260216_140405`
  - `outputs/shock_combo_refining_export/20260216_140437`
- Activated scenario control wiring in runtime coupled execution:
  - added `src/crm_model/scenario_controls.py`,
  - integrated control application into `src/crm_model/coupling.py` for SD inputs, SD->dMFA exchanges, and trade inputs,
  - added scenario control resolution reporting in `src/crm_model/run.py` (`scenario_controls_*` in `run_metadata.yml`, plus run-note summary).
- Verification outputs:
  - `outputs/baseline/20260216_133634`
  - `outputs/r_collection_push/20260216_133700`
- Corrected full-chain workflow visual semantics in:
  - `docs/diagrams/coupled_full_chain_scope.mmd`
  - `docs/WORKFLOW_VISUALS.md`
  - exogenous section now distinguishes file exogenous CSV inputs from runtime exchange variables generated during coupling.
- Closed indicator-assumption confirmations with user selections:
  - `1.A` keep `resource_depletion_time_years` as NA until `resources_kt` is populated,
  - `2.A` keep internal-boundary circularity input convention.
- Recorded confirmations in:
  - `configs/assumptions.yml`
  - `docs/ASSUMPTIONS_TO_CONFIRM.md`
  - `docs/DECISION_LOG.md`
- Implemented missing headline indicator formulas in `src/crm_model/indicators.py`:
  - `hhi_generic_0_1` (`sum_i share_i^2`)
  - `eol_rr_frac` (`old_scrap_recycled / old_scrap_generated`)
  - `eol_rir_frac` (`secondary_input_old_scrap / (primary_input + secondary_input_total)`)
  - `resource_depletion_time_years` (`resources / primary_production`)
- Extended dMFA/coupling outputs with explicit indicator input series so headline circularity indicators are computable from model outputs:
  - `old_scrap_recycled_kt_per_yr`
  - `primary_input_kt_per_yr`
  - `secondary_input_old_scrap_kt_per_yr`
  - `secondary_input_total_kt_per_yr`
- Reran baseline + scenario suite with updated indicator engine:
  - baseline: `outputs/baseline/20260216_124757`
  - scenarios: latest timestamps under each scenario folder at `~12:49:05..12:51:42` (2026-02-16)
- Rebuilt headline comparison table with updated indicator coverage:
  - `outputs/comparisons/20260216_125230/baseline_vs_scenarios_headline_indicators.csv`
  - `outputs/comparisons/20260216_125230/baseline_vs_scenarios_headline_indicators.md`
  - status summary: `66 ok`, `11 baseline_no_numeric_reporting;scenario_no_numeric_reporting` (all on `resource_depletion_time_years` due empty `resources_kt` data).
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
