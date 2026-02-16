# RISKS

## v5.3 MFASystem migration addendum

1. Process-network mapping risk (medium)
- The new full `MFASystem` stage graph maps existing equation outputs into explicit process-edge flows; mis-specified edge mapping could bias process-level diagnostics.
- Mitigation: inspect `process_flow_kt_per_yr` and run regression checks on headline indicators after each structural change.

## v5.2 cleanup addendum

1. Calibration-key migration risk (low)
- Active dMFA parameters now use `dmfa.calibration`; older local files may still use `dmfa.fallback_calibration`.
- Mitigation: runtime keeps legacy-key compatibility, but new edits should use `dmfa.calibration` only.

## v5.1 cleanup addendum

1. Backward-compatibility risk (medium)
- Configurations or scripts that relied on non-native execution modes (`native_if_available`, `fallback_only`) will now fail fast.
- Mitigation: keep `configs/coupling.yml` pinned to `native_required` and surface clear errors early in runs.

## v5.0 native-runtime risk addendum

1. Environment lock-in risk (high)
- Coupled full runs now require both BPTK-Py and flodym; missing dependencies will hard-fail execution.
- Mitigation: keep environment bootstrap documented and inspect `run_metadata.yml.runtime_modules` for every decision run.

2. Native-path regression risk (medium)
- dMFA native execution now uses full flodym-array tensor logic; any unnoticed indexing mismatch could affect mass-balance outputs.
- Mitigation: run baseline/scenario regression checks and keep calibration comparison snapshots versioned after each structural change.

## v4.10 native-dispatch risk addendum

Status:
- Superseded by v5.0 full native migration; retained for traceability.

1. Partial native-integration scope risk (medium)
- Native dispatch is active, but only selected dMFA array operations are currently delegated to flodym; this is not yet a full scientific reformulation in native frameworks.
- Mitigation: continue migration module-by-module, with side-by-side regression checks against agreed calibration targets.

2. Runtime dependency risk (medium)
- `native_required` mode will hard-fail runs if BPTK-Py/flodym are unavailable in the active environment.
- Mitigation (historical v4.10): this was previously handled via `native_if_available`; in v5.0 the active mitigation is strict environment management and metadata checks per run.

## v4.9 calibration-loop risk addendum

1. Objective weighting risk (high)
- Calibration ranking currently uses a TEMP weighted MAPE objective (`0.5*i_use + 0.5*stock`).
- Mitigation: treat weights as decision variables; rerun sweep if decision priority is stock-first or flow-first.

2. Parameter plausibility risk (high)
- Best coarse candidate increases `sd.capacity.utilization_target` to `1.05`, which can exceed strict physical utilization interpretation.
- Mitigation: interpret as effective fallback throughput factor, or constrain second-pass search to `<=1.0`.

3. Search granularity risk (medium)
- First sweep is coarse and may miss better local optima.
- Mitigation: run a narrowed second-pass around top candidates and optionally include additional knobs (e.g., new-scrap fraction).

4. Adoption-lag risk (medium)
- Previously open while waiting for adoption confirmation; now mitigated by applying the focused constrained candidate.
- Residual mitigation: keep calibration outputs/versioned so future re-tuning remains traceable.

## v4.9 trade-data risk addendum

1. HS concordance interpretation risk (high)
- HS92->HS22 correspondences can be many-to-many for some scrap-related headings.
- Mitigation: use official UNSD concordance, keep code baskets explicit, and track mapping artifacts in `data/reference/`.

2. Trade metric basis risk (high)
- OD weights currently use BACI trade value (`v`) rather than quantity (`q`).
- Mitigation: confirmed user choice (2026-02-15) to keep value-based OD weights; revisit only if quantity-centric calibration is introduced.

3. Classification boundary risk (medium)
- Inclusion of `800300` in tin refined basket can affect trade allocation and downstream indicators.
- Mitigation: confirmed user choice (2026-02-15) to include `800300` in `refined_metal`.

4. Scenario activation risk (high)
- Scenario levers/shocks are wired and unresolved placeholders are now TEMP-autofilled, but conservative defaults and sparse target series can still produce baseline-like outputs for some scenarios.
- Mitigation: check `run_metadata.yml` fields `scenario_controls_*` and scenario deltas together; replace autofill with explicit calibrated magnitudes in `configs/scenarios/*.yml` before decision use.

5. TEMP autofill interpretation risk (high)
- User-selected `1.B` enables conservative TEMP autofill magnitudes for unresolved controls; this improves runnability but is not a calibrated policy/shock package.
- Mitigation: expose autofill usage in run artifacts (`scenario_controls_autofill_*`), keep TEMP assumption explicit, and replace defaults with user-confirmed per-scenario magnitudes.

6. Stockpile proxy risk (medium)
- `stockpile_kt` / `stockpile_cover_years` currently use TEMP proxy fallback (`stock_refined_metal_kt`) when `stockpile_mass_kt` exogenous data is unavailable.
- Mitigation: populate `data/exogenous/stockpile_mass_kt.csv` with validated observations and retire TEMP proxy assumption.

## v4.8 risk register

1. Model-form risk (high)
- The SD and dMFA equations are fallback implementations intended to operationalize coupling, not final validated formulations.
- Mitigation: replace fallback equations with calibrated/validated formulations using the same variable contracts.

2. Data completeness risk (high)
- Required historic exogenous series are currently mostly empty templates.
- Mitigation: full runs now stop with explicit missing-input errors unless TEMP-approved.

3. Indicator coverage risk (medium)
- Headline formulas are now implemented for `hhi_generic_0_1`, `eol_rr_frac`, `eol_rir_frac`, and `resource_depletion_time_years`, but many non-headline indicators remain symbolic/unimplemented.
- Mitigation: continue priority implementation from `configs/indicators.yml`, focusing first on indicators used by decision dashboards.

4. Resource-data availability risk (high)
- `resource_depletion_time_years` requires `resources_kt`, which is currently empty (template-only), so the indicator remains NA in reporting outputs.
- Mitigation: user confirmed Option 1.A (keep NA, no proxy); next action is to populate `data/exogenous/resources_kt.csv` from validated sources.

5. Dependency/runtime risk (medium)
- `typer` may be unavailable in the active environment.
- Mitigation: CLI now has an argparse fallback for direct execution without `typer`.
