# TASK BOARD

## Goal
Build an executable baseline of the coupled SD + dMFA + OD-trade model using configuration-driven wiring.

## Definition of Done
- Coupled annual loop implemented in code (`SD -> smoothing -> dMFA -> OD trade`).
- Full run enforces stop-the-run when required historic inputs are missing/empty (unless TEMP-approved).
- Run artifacts include `configs_snapshot/`, `run_metadata.yml`, `assumptions_used.yml`, `indicators/`, and a short run note.
- Validators run successfully.

## Needed User Inputs
- To retire TEMP defaults: confirm explicit numeric magnitudes per scenario control in `configs/scenarios/*.yml`.

## Status
- Current phase: Scenario-control TEMP autofill (`1.B`) is active; unresolved placeholder magnitudes are resolved at runtime and explicitly tracked in run metadata/run notes.
- Native runtime migration is now completed and enforced for coupled runs (`native_required` for SD and dMFA in `configs/coupling.yml`).
- Fallback execution mode branches have been purged from SD/dMFA runtime code; execution path is now native-only.
- dMFA parameter naming has been normalized to `dmfa.calibration` (legacy `dmfa.fallback_calibration` remains readable for backward compatibility).
- dMFA native runtime now executes through an explicit full flodym `MFASystem` process network and exports `process_flow_kt_per_yr`.
- Latest outputs:
  - `outputs/calibration/baseline/20260216_091918`
  - `outputs/calibration/baseline/20260216_093512` (constrained detailed second pass)
  - `outputs/calibration/baseline/20260216_094654` (constrained quick-check sweep)
  - `outputs/calibration/baseline/20260216_100712` (focused constrained refinement)
  - `outputs/baseline/20260216_124757` (baseline rerun after headline indicator implementation)
  - `outputs/comparisons/20260216_125230` (updated baseline-vs-scenarios headline table)
  - `outputs/baseline/20260216_133634` (baseline rerun after scenario-control wiring)
  - `outputs/r_collection_push/20260216_133700` (example run showing unresolved control reporting)
  - `outputs/baseline/20260216_140102` (baseline with autofill-enabled metadata fields)
  - `outputs/r_collection_push/20260216_140405` (autofill-applied lever run)
  - `outputs/shock_combo_refining_export/20260216_140437` (autofill-applied compound shock run)
  - `outputs/baseline/20260216_153435` (baseline after missing-policy + demand fallback fix)
  - `outputs/comparisons/20260216_153822` (full latest baseline-vs-scenarios headline comparison)
  - `outputs/r_collection_push/20260216_192200` (headline-gap closure rerun; updated headline set now numeric)
  - `outputs/baseline/20260216_203603` (baseline rerun with native dispatch metadata)
  - `outputs/r_collection_push/20260216_203642` (scenario rerun with native dispatch metadata)
  - `outputs/baseline/20260216_204820` (full native-migrated baseline rerun)
  - `outputs/r_collection_push/20260216_204635` (full native-migrated scenario rerun)
  - `outputs/baseline/20260216_205554` (post-cleanup native-only baseline rerun)
  - `outputs/r_collection_push/20260216_205632` (post-cleanup native-only scenario rerun)
  - `outputs/baseline/20260216_210400` (native-only baseline rerun after code/doc cleanup)
  - `outputs/r_collection_push/20260216_210435` (native-only scenario rerun after code/doc cleanup)
  - `outputs/baseline/20260216_213718` (baseline after full flodym MFASystem process-network migration)
  - `outputs/r_collection_push/20260216_213814` (scenario check after full flodym MFASystem process-network migration)
- Next phase: replace TEMP autofill defaults with explicit scenario magnitudes, replace TEMP stockpile proxy with validated `stockpile_mass_kt` exogenous data, and run focused regression/calibration checks on the new native runtime path.
