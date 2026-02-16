# Scenarios (configs/scenarios)

A scenario is a declarative patch over:
- exogenous inputs (CSV series) and/or
- parameter files and/or
- time-varying policy levers and shocks.

Conventions:
- Scenarios may inherit from a baseline scenario.
- Any `null` under levers/shocks is a hard stop: ask the user (no-guess).
- Use simple time shapes:
  - ramp: linear change start→end
  - step: new level at year
  - pulse: temporary multiplier for a duration

Variables referenced should exist in `registry/variable_registry.yml` and be wired in `configs/coupling.yml`.
