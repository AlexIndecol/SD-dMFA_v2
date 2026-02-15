# DECISION LOG

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
