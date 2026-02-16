# ASSUMPTIONS

This document mirrors `configs/assumptions.yml` for stakeholder-readable context.

## TEMP discipline
- TEMP assumptions must be explicitly declared in `configs/assumptions.yml`.
- TEMP assumptions used in a run are exported to `outputs/<scenario>/<timestamp>/assumptions_used.yml`.
- TEMP assumptions should include an explicit replacement plan and owner confirmation path.

## Current state
- Active TEMP assumptions:
  - `scenario_control_autofill_conservative_temp`
    - Scope: scenario control magnitudes.
    - Implementation: `configs/scenario_autofill.yml` (`enabled: true`).
    - Why TEMP: conservative defaults are placeholders, not calibrated policy/shock values.
    - Replacement path: replace control magnitudes directly in `configs/scenarios/*.yml` and disable autofill.
  - `stockpile_proxy_from_refined_buffer_temp`
    - Scope: stockpile indicators.
    - Implementation: `src/crm_model/indicators.py` fallback resolver.
    - Why TEMP: no validated `stockpile_mass_kt` exogenous dataset is currently provided.
    - Replacement path: populate `data/exogenous/stockpile_mass_kt.csv` and disable proxy fallback.
