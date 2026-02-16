# Assumptions to confirm

Open confirmation (new):

1) Stockpile indicator data source (medium impact)
- Current TEMP implementation: use `stock_refined_metal_kt` as proxy for `stockpile_kt` / `stockpile_cover_years` when `stockpile_mass_kt` is unavailable.
- Confirmation options:
  - A) keep TEMP proxy until validated `stockpile_mass_kt` is sourced.
  - B) require explicit `stockpile_mass_kt` and keep stockpile indicators NA until then.

All previously open assumptions have been resolved:

1) Missing-data handling outside the historic calibration window (accepted)
- before_first_year: hold_first
- after_last_year: hold_last
- inside_gaps: interpolate_linear

2) Overlap year 2020 (resolved: **no overlap**)
- historic: 1870–2019 (inclusive)
- reporting: 2020–2100 (inclusive)

If you later want per-variable missing policies (e.g., GAS=0 before first year), you can extend `configs/data_sources.yml`
to allow per-variable overrides.

## v4.9 trade ingestion confirmations (resolved on 2026-02-15)

1) BACI OD weighting basis (high impact)
- Selected: **Option A** (keep OD weights based on BACI trade value `v`, thousand USD).

2) Tin refined basket boundary in HS22 (high impact)
- Selected: **Option A** (include `800300` in `refined_metal`).

3) HS92 to HS22 stitching rule (medium impact)
- Selected: **Option A** (HS92 for `<2022`, HS22 for `>=2022`, no chain-link scaling).

## v4.9 calibration confirmations (resolved on 2026-02-16)

Selections provided by user:
- `1.A` keep equal objective weights (`0.5*i_use + 0.5*stock`).
- `2.B` constrain fallback calibration to `utilization_target <= 1.0`.
- `3.B` keep current baseline parameters unchanged and run a second-pass calibration first.

Executed second-pass runs:
- constrained detailed sweep:
  - `outputs/calibration/baseline/20260216_093512`
  - grid: `util={0.90,0.95,1.00}`, `eol_mult={0.95,1.00,1.05}`, `ref_exp={0.13,0.15,0.17}`, `new_scrap={0.04,0.05,0.06}`
  - best candidate:
    - `sd.capacity.utilization_target = 1.00`
    - `dmfa.fallback_calibration.eol_outflow_multiplier = 1.00`
    - `dmfa.fallback_calibration.new_scrap_fraction_of_demand = 0.06`
    - `dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal = 0.17`
  - best objective (weighted MAPE): `0.222653`
- constrained quick-check sweep:
  - `outputs/calibration/baseline/20260216_094654`
  - fixed `new_scrap=0.05` gave consistent structure and best objective `0.224184`.

Performance context (using detailed constrained sweep best):
- vs current default baseline candidate (`candidate_id=14` in first sweep):
  - objective improvement: about `10.10%`
  - `i_use` MAPE improvement: about `12.99%`
  - stock MAPE improvement: about `8.26%`
- vs unconstrained first-pass best (`candidate_id=33`, util `1.05`):
  - constrained best is about `0.84%` worse on objective.

## v4.9 focused constrained sweep (executed after user selected Option B)

Executed:
- `outputs/calibration/baseline/20260216_100712`
- fixed constraints:
  - `utilization_target = 1.0`
  - `eol_outflow_multiplier = 1.0`
- focused grid:
  - `refined_export_cap_fraction = {0.15, 0.16, 0.17, 0.18, 0.19}`
  - `new_scrap_fraction_of_demand = {0.05, 0.055, 0.06, 0.065, 0.07}`

Focused best candidate:
- `sd.capacity.utilization_target = 1.00`
- `dmfa.fallback_calibration.eol_outflow_multiplier = 1.00`
- `dmfa.fallback_calibration.new_scrap_fraction_of_demand = 0.07`
- `dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal = 0.19`
- objective (weighted MAPE): `0.220885`

Performance context:
- vs current default baseline candidate:
  - objective improvement: about `10.81%`
- vs prior constrained best (`outputs/calibration/baseline/20260216_093512`):
  - objective improvement: about `0.79%`
- vs unconstrained first-pass best (`utilization_target=1.05`):
  - objective gap: about `0.04%` (focused constrained is very close)

## v4.9 adoption confirmation (resolved on 2026-02-16)

User selections sequence:
- first: `Option B` (keep baseline unchanged, run focused sweep),
- then: `Option A` (adopt focused constrained best).

Applied baseline parameter updates:
- `configs/parameters/parameters_sd.yml`
  - `sd.capacity.utilization_target = 1.0`
- `configs/parameters/parameters_dmfa.yml`
  - `dmfa.fallback_calibration.eol_outflow_multiplier = 1.0`
  - `dmfa.fallback_calibration.new_scrap_fraction_of_demand = 0.07`
  - `dmfa.fallback_calibration.export_cap_fraction_by_commodity.refined_metal = 0.19`

Verification run:
- full baseline rerun output:
  - `outputs/baseline/20260216_101506`
- achieved historic objective (equal weights):
  - `0.220885` (`i_use` MAPE `0.165619`, stock MAPE `0.276151`)
  - consistent with focused sweep best (`outputs/calibration/baseline/20260216_100712`).

Open calibration confirmations:
- none at this stage.

## v4.9 indicator confirmations (resolved on 2026-02-16)

Selections provided by user:
- `1.A` keep `resource_depletion_time_years` as NA until `resources_kt` is populated.
- `2.A` keep current internal-boundary convention for circularity inputs (`primary_input_*`, `secondary_input_*`, `old_scrap_recycled_*`).

Implications:
- `resource_depletion_time_years` remains non-numeric in current outputs until `data/exogenous/resources_kt.csv` is filled with numeric values.
- Circularity headline indicators (`eol_rr_frac`, `eol_rir_frac`) continue using the fallback dMFA internal-boundary mapping currently implemented in engine outputs.

Open confirmations:
- none at this stage.

## v4.9 scenario control calibration confirmation (resolved on 2026-02-16)

Selection provided by user:
- `1.B` enable a TEMP conservative auto-fill package for unresolved scenario magnitudes.

Applied implementation:
- Added `configs/scenario_autofill.yml` with conservative defaults for unresolved controls.
- Wired autofill into control resolution and runtime application in:
  - `src/crm_model/scenario_controls.py`
  - `src/crm_model/coupling.py`
  - `src/crm_model/run.py`
- Extended run artifacts to expose autofill status:
  - `run_metadata.yml`: `scenario_controls_autofill_enabled`, `scenario_controls_autofill_used`
  - `run_note.md`: one-line autofill status in scenario-control summary.

Governance:
- Added TEMP assumption `scenario_control_autofill_conservative_temp` in `configs/assumptions.yml`.
- Replacement path remains explicit: migrate to user-confirmed numeric magnitudes in `configs/scenarios/*.yml` and disable autofill.

Open confirmations:
- none at this stage.
