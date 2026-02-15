# AGENTS.md — Compact working rules

This repo is built to be **usable by non-programmers** and **reproducible for modellers**.  
**Never hide assumptions. Never guess high-impact inputs.**

---

## 1) Repo contract (single sources of truth)
- **Inputs:** `data/exogenous/<variable>.csv` (one file per variable, long format, `value` column).
- **Variables:** must exist in `registry/variable_registry.yml`.
- **Codes/dimensions:** `configs/dimensions.yml`.
- **Time windows:** `configs/time.yml` (spinup / historic / reporting).
- **Coupling wiring:** `configs/coupling.yml` (no ad-hoc couplings in code).
- **Indicators:** `configs/indicators.yml` (no “extra” indicators outside this file).
- **j vs jd:** `configs/end_use_detail_mapping.yml` + share constraints.

**Stop-the-run rule:** If required historic inputs are missing/empty, **stop and ask the user**, unless explicitly declared TEMP in `configs/assumptions.yml`.

---

## 2) No-guess policy (ask often)
Ask the user for anything that can change results materially:
- exogenous/endogenous boundaries, units, mappings, calibration targets, scenarios, lifetimes, trade interpretation.
If you must proceed temporarily: declare as **TEMP**, minimize impact, and make it visible in outputs.

---

## 3) Configuration over hardcoding
Prefer changes in:
- `configs/*.yml`, `data/exogenous/*.csv`, `docs/*`, `notebooks/*`, `docs/diagrams/*`.
Avoid hardcoding materials/regions/end-uses/stages in code.

---

## 4) Reproducibility (minimum run artifacts)
Every run writes to `outputs/<scenario>/<timestamp>/`:
- `configs_snapshot/`, `run_metadata.yml`, `assumptions_used.yml`, `indicators/`
- (recommended) `data_digest.yml`

A run is “done” only if:
- validator passes (`scripts/validate_exogenous_inputs.py`)
- assumptions used are recorded (TEMP flagged)
- a short run note exists (decision question + interpretation limits)

---

## 5) Tasks, decisions, risks
Keep these up to date:
- `docs/TASK_BOARD.md` (goal, DoD, needed user inputs, status)
- `docs/DECISION_LOG.md` (what decision the work supports)
- `docs/RISKS.md` (limitations/risks per version + mitigation)
- `docs/CHANGELOG.md` (what changed + why)

---

## 6) TEMP assumptions (mandatory discipline)
- Declare in `configs/assumptions.yml` + explain in `docs/ASSUMPTIONS.md`.
- Export into each run’s `assumptions_used.yml`.
- Never let TEMP become permanent without confirmation.

---

## 7) Schema/version hygiene
Any change to vocabularies, variable names, or CSV schemas requires:
- version bump + changelog entry
- regenerated templates/docs
- validator update
- diagram regeneration if coupling changed

---

## 8) Keep it clean & inspectable
- Prefer notebooks and diagrams for stakeholder inspection.
- Remove/depurate unused files; deprecate explicitly when unsure.
- Keep notebooks runnable and Mermaid diagrams in sync with configs.

---
## 9) Do / Don’t
**Do:** ask, configure, validate, snapshot, document.  
**Don’t:** guess, silently default, hardcode dimension members, or report uncodified indicators.
