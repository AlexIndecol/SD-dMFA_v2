# DECISION LOG

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
