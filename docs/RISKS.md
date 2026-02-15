# RISKS

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

## v4.8 risk register

1. Model-form risk (high)
- The SD and dMFA equations are fallback implementations intended to operationalize coupling, not final validated formulations.
- Mitigation: replace fallback equations with calibrated/validated formulations using the same variable contracts.

2. Data completeness risk (high)
- Required historic exogenous series are currently mostly empty templates.
- Mitigation: full runs now stop with explicit missing-input errors unless TEMP-approved.

3. Indicator coverage risk (medium)
- Only a subset of indicators is explicitly implemented in `src/crm_model/indicators.py`; others output note files.
- Mitigation: implement formulas incrementally in priority order from `configs/indicators.yml`.

4. Dependency/runtime risk (medium)
- `typer` may be unavailable in the active environment.
- Mitigation: CLI now has an argparse fallback for direct execution without `typer`.
