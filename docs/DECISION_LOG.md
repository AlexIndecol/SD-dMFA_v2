# DECISION LOG

## 2026-02-16 - Migrate dMFA native runtime to full flodym MFASystem process network
Decision supported:
- Replace array-only flodym runtime usage with explicit process-network execution in dMFA.

Decision:
- Implement full stage-process MFASystem wiring in `src/crm_model/dmfa_model.py`:
  - explicit process nodes (`sysenv` + configured stages),
  - explicit named stage/trade flows via `FlowDefinition`,
  - per-year process-flow export (`process_flow_kt_per_yr`).
- Keep existing indicator-facing output contracts unchanged while adding process-network diagnostics.
- Extend coupled outputs with `process_flow_kt_per_yr` in `src/crm_model/coupling.py`.
- Verification runs:
  - `outputs/baseline/20260216_213718`
  - `outputs/r_collection_push/20260216_213814`

Rationale:
- Aligns runtime implementation with requested native framework semantics (full MFASystem process graph rather than array-only tensor arithmetic).

Limits:
- The current MFASystem network keeps the existing equation logic and data contracts; it is a structural/runtime migration, not yet a full scientific recalibration of all stage equations.

## 2026-02-16 - Post-purge cleanup and calibration-key normalization
Decision supported:
- Complete the requested cleanup after fallback execution-mode retirement.

Decision:
- Keep native runtime enforcement unchanged (`execution_mode = native_required` for SD/dMFA).
- Normalize active dMFA parameter naming to `dmfa.calibration` while keeping legacy `dmfa.fallback_calibration` readable in code for backward compatibility.
- Normalize calibration tooling to the same key (`scripts/calibration/run_baseline_calibration_loop.py` now writes `dmfa.calibration.*`).
- Remove minor no-op/formatting clutter in coupling runtime code.
- Verification runs:
  - `outputs/baseline/20260216_210400`
  - `outputs/r_collection_push/20260216_210435`

Rationale:
- Clarifies current runtime contract and reduces ambiguity in parameter naming after migration.

Limits:
- Historical docs and artifacts keep legacy naming where they reflect past runs/decisions.

## 2026-02-16 - Purge fallback execution mode and code cleanup
Decision supported:
- Remove residual fallback execution-mode branches after native migration.

Decision:
- Restrict SD and dMFA execution-mode validation to `native_required` only.
- Remove legacy dMFA mixed fallback/native runtime path and retain only native flodym-array execution.
- Update run metadata note to explicitly indicate fallback execution removal.
- Verification runs:
  - `outputs/baseline/20260216_205554`
  - `outputs/r_collection_push/20260216_205632`

Rationale:
- Reduces runtime ambiguity and maintenance burden by ensuring a single executable path per module.

Limits:
- This cleanup does not retire TEMP data assumptions or calibration naming legacy in historical documentation.

## 2026-02-16 - Full migration to native runtime execution
Decision supported:
- Complete migration from mixed native+fallback execution to strict native execution in coupled annual runs.

Decision:
- Enforce native runtime on both major modules:
  - SD: BPTK-Py required at runtime.
  - dMFA: flodym required at runtime.
- Disable runtime fallback execution modes in model builders (`fallback_only` now fails fast).
- Set coupling defaults to strict native mode:
  - `configs/coupling.yml -> modules.sd.execution_mode = native_required`
  - `configs/coupling.yml -> modules.dmfa.execution_mode = native_required`
- Implement full flodym-array computation path for dMFA native execution.
- Verification runs:
  - `outputs/baseline/20260216_204820`
  - `outputs/r_collection_push/20260216_204635`

Rationale:
- Ensures model behavior is executed through the declared native frameworks rather than fallback equations, aligning runtime with framework governance expectations.

Limits:
- Indicator-level TEMP assumptions (e.g., stockpile proxy and scenario autofill) remain unchanged; this decision addresses execution framework migration, not assumption retirement.

## 2026-02-16 - Activate native framework wiring in coupled runtime
Decision supported:
- Move from fallback-only runtime wiring to explicit native-framework dispatch while preserving deterministic fallback paths.

Decision:
- Add execution-mode contract for SD and dMFA modules:
  - supported modes: `native_required`, `native_if_available`, `fallback_only`.
- Wire SD runtime to dispatch via BPTK-Py when available.
- Wire dMFA runtime to dispatch via flodym-backed array computations when available.
- Add runtime transparency in run artifacts:
  - `run_metadata.yml.runtime_modules` now reports requested engine, execution mode, native availability, and actual `use_native` flag per module.
- Set coupling defaults to native-when-available:
  - `configs/coupling.yml -> modules.sd.execution_mode = native_if_available`
  - `configs/coupling.yml -> modules.dmfa.execution_mode = native_if_available`
- Verification runs:
  - `outputs/baseline/20260216_203603`
  - `outputs/r_collection_push/20260216_203642`

Rationale:
- Keeps the same variable contracts and fallback resilience while enabling immediate use of native SD/dMFA frameworks in environments where dependencies are present.

Limits:
- Current native integration is dispatch-level with selected flodym-backed array operations; scientific equation forms are still the configured deterministic coupled equations and require separate calibration/validation work.

## 2026-02-16 - Fill headline data gaps with explicit fallback assumptions and rerun latest scenario
Decision supported:
- Compute updated headline set in `configs/reporting.yml` without introducing silent defaults.

Decision:
- Implement indicator-engine support for:
  - `apparent_consumption_kt_per_yr`
  - `stockpile_kt`
  - `stockpile_cover_years`
- Extend coupled outputs so apparent-consumption components are available:
  - `domestic_production_kt_per_yr`
  - `stock_change_kt_per_yr`
- Apply TEMP stockpile proxy fallback:
  - if `stockpile_mass_kt` is missing, use `stock_refined_metal_kt` as proxy for stockpile indicators.
- Record TEMP governance in `configs/assumptions.yml` (`stockpile_proxy_from_refined_buffer_temp`).
- Rerun latest requested scenario:
  - `outputs/r_collection_push/20260216_192200`

Rationale:
- Previous headline failures were primarily due to indicator pipeline gaps and one missing stockpile input series.
- The selected fallback keeps scenario sensitivity (proxy uses modelled refined stock) while making assumptions explicit.

Limits:
- Stockpile-related headline values are provisional until `data/exogenous/stockpile_mass_kt.csv` is populated with validated observations.

## 2026-02-16 - Fix zero-demand reporting path (missing-policy + GAS numeric gating)
Decision supported:
- Resolve the observed issue where reporting-period `demand_kt_per_yr` stayed at zero despite configured scenario controls.

Decision:
- Implement exogenous missing-policy application in loader (`src/crm_model/io_exogenous.py`) using `configs/data_sources.yml`.
- Update SD demand branching (`src/crm_model/sd_model.py`) to use observed GAS only when numeric values exist in the year slice.
- Rerun baseline and full scenario suite under corrected demand path:
  - baseline: `outputs/baseline/20260216_153435`
  - comparison: `outputs/comparisons/20260216_153822`

Rationale:
- Previously, GAS rows existed for reporting years but with all-NaN `value`; grouped sum converted these to zeros, forcing demand to zero.

Limits:
- Missing-policy currently applies globally by variable file; per-variable override granularity remains a future enhancement.

## 2026-02-16 - Apply scenario-control TEMP autofill package (user selection 1.B)
Decision supported:
- Resolve unresolved scenario control placeholders without silently inventing permanent scenario magnitudes.

Decision:
- Enable conservative TEMP autofill defaults in new config `configs/scenario_autofill.yml`.
- Wire autofill into control resolution and runtime control application via:
  - `src/crm_model/scenario_controls.py`
  - `src/crm_model/coupling.py`
  - `src/crm_model/run.py`
- Extend run artifacts:
  - `run_metadata.yml`: `scenario_controls_autofill_enabled`, `scenario_controls_autofill_used`
  - `run_note.md`: include autofill status in control summary
- Record TEMP governance in `configs/assumptions.yml` (`scenario_control_autofill_conservative_temp`).
- Verification runs:
  - `outputs/baseline/20260216_140102`
  - `outputs/r_collection_push/20260216_140405`
  - `outputs/shock_combo_refining_export/20260216_140437`

Rationale:
- Keeps the no-guess policy visible while allowing scenario execution and explicit tracking of temporary filled magnitudes.

Limits:
- Autofill defaults are placeholders and not calibrated policy/shock assumptions; explicit per-scenario magnitudes remain required for decision-grade interpretation.

## 2026-02-16 - Activate scenario control wiring in coupled run path
Decision supported:
- Execute the next workflow phase to ensure scenario levers/shocks can propagate into SD+dMFA+trade dynamics when configured.

Decision:
- Implement scenario-control application in runtime for:
  - SD inputs from exogenous (`gas_to_use_observed_kt_per_yr`, `capacity_stage_observed_kt_per_yr`, `use_share_j_frac`)
  - SD runtime outputs/exchanges (`demand_kt_per_yr`, `collection_rate_0_1`, `recovery_rate_0_1`, `recycling_yield_0_1`, `lifetime_multiplier_ge_1`, `capacity_stage_raw_kt_per_yr`, `price_index_rel`)
  - Trade inputs (`od_preference_weight_0_1`, `trade_factor_i_ge_1`)
- Add scenario control resolution summary to run artifacts:
  - `run_metadata.yml`: `scenario_controls_total`, `scenario_controls_resolved`, `scenario_controls_unresolved`
  - `run_note.md`: one-line control summary
- Verify with reruns:
  - baseline: `outputs/baseline/20260216_133634`
  - scenario sample: `outputs/r_collection_push/20260216_133700`

Rationale:
- Removes the prior wiring gap where scenario definitions were loaded but not used in annual model computations.

Limits:
- Many scenario files still use placeholder `null` magnitudes, so user calibration/confirmation is required before scenario deltas can be interpreted.

## 2026-02-16 - Confirm indicator assumptions 1.A and 2.A
Decision supported:
- Close open assumption confirmations introduced after headline indicator implementation.

Decision:
- Confirm `1.A`: keep `resource_depletion_time_years` as NA until `resources_kt` is populated with numeric data.
- Confirm `2.A`: keep current internal-boundary mapping for circularity indicator inputs (`primary_input_kt_per_yr`, `secondary_input_total_kt_per_yr`, `secondary_input_old_scrap_kt_per_yr`, `old_scrap_recycled_kt_per_yr`).
- Record both confirmations in `configs/assumptions.yml` and close the open items in `docs/ASSUMPTIONS_TO_CONFIRM.md`.

Rationale:
- Avoids introducing high-impact proxies without evidence while preserving consistent circularity accounting in the current fallback boundary.

Limits:
- `resource_depletion_time_years` remains non-numeric until `data/exogenous/resources_kt.csv` is filled.

## 2026-02-16 - Headline indicator engine completion (option 2 execution)
Decision supported:
- Make baseline-vs-scenario headline comparison interpretable by implementing previously missing headline formulas.

Decision:
- Implement formulas in `src/crm_model/indicators.py` for:
  - `hhi_generic_0_1`
  - `eol_rr_frac`
  - `eol_rir_frac`
  - `resource_depletion_time_years`
- Extend fallback dMFA/coupling outputs with explicit indicator input series:
  - `old_scrap_recycled_kt_per_yr`
  - `primary_input_kt_per_yr`
  - `secondary_input_old_scrap_kt_per_yr`
  - `secondary_input_total_kt_per_yr`
- Rerun baseline and full scenario suite with updated engine and regenerate comparison artifact:
  - baseline: `outputs/baseline/20260216_124757`
  - comparison: `outputs/comparisons/20260216_125230`

Rationale:
- Removes “formula not implemented” / “missing required inputs” blockers for 3 of 4 missing headline indicators and restores direct comparability in scenario tables.

Limits:
- `resource_depletion_time_years` remains non-numeric because `resources_kt` exogenous data is currently empty; user confirmation needed on data/fallback policy.

## 2026-02-16 - Adopt focused constrained calibration candidate into baseline
Decision supported:
- Finalize baseline fallback calibration parameters after focused constrained refinement.

Decision:
- Adopt focused constrained best candidate from `outputs/calibration/baseline/20260216_100712`:
  - `sd.capacity.utilization_target = 1.0`
  - `dmfa.fallback_calibration.eol_outflow_multiplier = 1.0`
  - `dmfa.fallback_calibration.new_scrap_fraction_of_demand = 0.07`
  - `dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal = 0.19`
- Rerun full baseline with adopted parameters:
  - `outputs/baseline/20260216_101506`

Rationale:
- Preserves user-selected physical cap (`utilization <= 1.0`) while reaching near-unconstrained fit quality.

Limits:
- Adoption remains within fallback-equation model form; scientific-structure calibration remains future work.

## 2026-02-16 - Focused constrained calibration refinement
Decision supported:
- Execute one more focused calibration sweep under confirmed constraints (`utilization_target <= 1.0`, equal objective weights) before any parameter adoption.

Decision:
- Run focused sweep with fixed:
  - `sd.capacity.utilization_target = 1.0`
  - `dmfa.fallback_calibration.eol_outflow_multiplier = 1.0`
- Sweep:
  - `dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal` in `{0.15,0.16,0.17,0.18,0.19}`
  - `dmfa.fallback_calibration.new_scrap_fraction_of_demand` in `{0.05,0.055,0.06,0.065,0.07}`
- Best focused constrained candidate (`outputs/calibration/baseline/20260216_100712`):
  - `utilization_target = 1.00`
  - `eol_outflow_multiplier = 1.00`
  - `refined_export_cap_fraction = 0.19`
  - `new_scrap_fraction_of_demand = 0.07`
  - objective = `0.220885`

Rationale:
- Tightened search around constrained optimum improves fit while preserving the user-imposed utilization cap.

Limits:
- Baseline parameters are still intentionally unchanged pending explicit adoption confirmation.

## 2026-02-16 - Constrained second-pass fallback calibration (user-selected)
Decision supported:
- Apply user-selected calibration governance (`1.A`, `2.B`, `3.B`) and identify the best constrained fallback candidate before any baseline parameter adoption.

Decision:
- Keep equal objective weights (`0.5*i_use + 0.5*stock`).
- Enforce calibration constraint `utilization_target <= 1.0`.
- Run second-pass calibration without changing baseline parameters yet.
- Constrained second-pass best candidate (from `outputs/calibration/baseline/20260216_093512`):
  - `sd.capacity.utilization_target = 1.00`
  - `dmfa.fallback_calibration.eol_outflow_multiplier = 1.00`
  - `dmfa.fallback_calibration.new_scrap_fraction_of_demand = 0.06`
  - `dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal = 0.17`

Rationale:
- Respects physical-interpretability preference (`utilization <= 1.0`) while preserving measurable calibration gains versus default fallback settings.

Limits:
- Baseline parameter files are intentionally not updated yet; adoption remains a separate confirmation step.

## 2026-02-16 - First-pass fallback calibration sweep
Decision supported:
- Identify candidate fallback parameter sets that improve historic fit to `gas_to_use_observed_kt_per_yr` and `in_use_stock_observed_kt`.

Decision:
- Run a coarse grid sweep over:
  - `sd.capacity.utilization_target`
  - `dmfa.fallback_calibration.eol_outflow_multiplier`
  - `dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal`
  - `dmfa.fallback_calibration.new_scrap_fraction_of_demand` (held at 0.05 in this first pass)
- Score candidates on weighted MAPE (`0.5*i_use + 0.5*stock`) over historic years 1870–2019.

Rationale:
- Provides a reproducible, explicit starting point for calibration without silently changing baseline parameters.

Limits:
- Coarse search only; objective weighting and acceptance thresholds are TEMP and require user confirmation before adopting parameter overrides.
- Best candidate in this pass uses `utilization_target=1.05`, which may require interpretation as an effective throughput factor rather than strict physical utilization.

## 2026-02-15 - BACI HS92/HS22 linked OD baseline
Decision supported:
- Build a consistent BACI-based trade time series and populate baseline OD exogenous weights.

Decision:
- Use official UNSD `HS-SITC-BEC Correlations_2022` concordance to bridge HS92 and HS22 code systems.
- Construct OD weights with BACI bilateral flows using HS92 for years < 2022 and HS22 for years >= 2022.
- Keep pre-1995 OD years on explicit fallback (gravity-style from BGS import/export marginals).
- Normalize mass quantities from source tonnes to model `kt` for BGS and OWID series before writing exogenous historic files.

Rationale:
- This uses an authoritative concordance source while preserving continuity and coverage across the model horizon.
- BACI provides bilateral structure needed by OD weights that BGS aggregate trade cannot provide.

Limits:
- Confirmed by user on 2026-02-15: Option A for all three choices (value-based OD weights, include `800300` in tin refined basket, and no chain-link scaling).

## 2026-02-15 - First runnable coupled baseline
Decision supported:
- Start implementing a concrete SD+dMFA+trade engine instead of keeping interface-only skeletons.

Decision:
- Implement a deterministic fallback annual coupled engine that uses existing configs and exogenous contracts.
- Enforce a hard stop for full runs when required historic exogenous inputs are missing/empty unless TEMP-approved in `configs/assumptions.yml`.

Rationale:
- This enables immediate end-to-end execution semantics while preserving governance and reproducibility rules.
- It avoids hidden defaults for high-impact historic calibration inputs.

Limits:
- Equations are fallback implementations for coupling workflow bootstrapping, not final calibrated scientific equations.
