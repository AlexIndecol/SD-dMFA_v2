# Config index

- dimensions.yml: vocabularies and dimension codes
- time.yml: model horizon + period split + coupling stabilizer controls
- data_sources.yml: exogenous ingestion policy (one CSV per variable)
- coupling.yml: explicit SD↔dMFA↔trade exchange contract (what passes where)
- indicators.yml: machine-readable indicator definitions
- parameters/: model parameters

- assumptions.yml: assumption registry (TEMP + confirmed)
- reporting.yml: reporting contract (headline indicators, exports, plots)
- scenarios/: scenario blueprints and registry

Validation:
- scripts/validate_consistency.py: checks indicator requirements and scenario registry integrity.
