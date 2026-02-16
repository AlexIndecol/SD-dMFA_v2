# TASK BOARD

## Goal
Build an executable baseline of the coupled SD + dMFA + OD-trade model using configuration-driven wiring.

## Definition of Done
- Coupled annual loop implemented in code (`SD -> smoothing -> dMFA -> OD trade`).
- Full run enforces stop-the-run when required historic inputs are missing/empty (unless TEMP-approved).
- Run artifacts include `configs_snapshot/`, `run_metadata.yml`, `assumptions_used.yml`, `indicators/`, and a short run note.
- Validators run successfully.

## Needed User Inputs
- None currently open for baseline calibration adoption.

## Status
- Current phase: Baseline fallback calibration adopted and verified.
- Latest outputs:
  - `outputs/calibration/baseline/20260216_091918`
  - `outputs/calibration/baseline/20260216_093512` (constrained detailed second pass)
  - `outputs/calibration/baseline/20260216_094654` (constrained quick-check sweep)
  - `outputs/calibration/baseline/20260216_100712` (focused constrained refinement)
  - `outputs/baseline/20260216_101506` (baseline rerun after focused candidate adoption)
- Next phase: proceed to scenario testing/sensitivity analysis using adopted baseline calibration settings.
