# Changelog

## v4.7
- Confirmed modelling decisions in configs/assumptions.yml:
  - OD matrices are **weights** (relative preferences), not shares.
  - Trade commodities are **concentrate**, **scrap**, **refined_metal**.
  - Reuse/reman is represented via **lifetime extension** (lognormal mu shift by ln(k)).
  - Demand-based indicators use **realized demand = I_use**.
- Added SD/dMFA/trade/coupling **module interfaces** in `src/crm_model/` (config-driven; equations remain to be implemented).
- Implemented a simple OD **weights allocator** (`allocate_od_weights`) supporting commodity-specific export caps.
- Added accounting scaffolding for **commodity buffers**, **balancing items**, and **scrap split/delay** (variables + coupling contract).
- Extended indicator config with v4.7 add-ons: **UR**, demand-based DIR (realized), balancing diagnostics, and buffer coverage.
- Added new Jupyter cookbooks (`notebooks/07–09`) and updated Mermaid workflow to reflect buffers/balancing.


## v4.6
- Added scripts/validate_consistency.py (indicator requires + scenario registry checks).
- Added docs/CONSISTENCY_SCAN_v4_6.md and auto-added placeholder variables to ensure internal consistency.


## v4.5
- Parsed INDICATORS.md into configs/indicators.yml (45 indicators/metrics + inferred dims/requires).
- Updated docs/INDICATORS_SPEC.md to reflect auto-parsing and conventions.


## v4.4
- Added AGENTS.md (compact governance rules).
- Added configs/assumptions.yml and configs/reporting.yml.
- Added scenario blueprints for R-strategies and disruptions (configs/scenarios/*).
- Added combined policy package and compound shock scenario blueprints.
