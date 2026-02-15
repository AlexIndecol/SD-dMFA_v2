# Assumptions to confirm

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
