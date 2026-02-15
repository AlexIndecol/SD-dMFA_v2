# TASK BOARD

## Goal
Build an executable baseline of the coupled SD + dMFA + OD-trade model using configuration-driven wiring.

## Definition of Done
- Coupled annual loop implemented in code (`SD -> smoothing -> dMFA -> OD trade`).
- Full run enforces stop-the-run when required historic inputs are missing/empty (unless TEMP-approved).
- Run artifacts include `configs_snapshot/`, `run_metadata.yml`, `assumptions_used.yml`, `indicators/`, and a short run note.
- Validators run successfully.

## Needed User Inputs
- None for baseline exogenous population task (v4.9 confirmations closed).

## Status
- Current phase: Required historic exogenous baseline inputs populated and validated.
- Next phase: proceed to calibration/testing runs.
