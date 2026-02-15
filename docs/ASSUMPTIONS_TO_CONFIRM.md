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

## v4.9 trade ingestion confirmations (resolved on 2026-02-15)

1) BACI OD weighting basis (high impact)
- Selected: **Option A** (keep OD weights based on BACI trade value `v`, thousand USD).

2) Tin refined basket boundary in HS22 (high impact)
- Selected: **Option A** (include `800300` in `refined_metal`).

3) HS92 to HS22 stitching rule (medium impact)
- Selected: **Option A** (HS92 for `<2022`, HS22 for `>=2022`, no chain-link scaling).
