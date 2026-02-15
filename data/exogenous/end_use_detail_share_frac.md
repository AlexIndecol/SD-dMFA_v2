# end_use_detail_share_frac.csv (optional)

Shares allocating 7 end-use buckets (j) into end-use detail categories (jd) for CRM tracking.
- Columns: t,r,m,j,jd,value
- Unit: 0..1
- Constraint: sum over jd must equal 1 for each (t,r,m,j).
- Only rows where jd maps to j are included (see configs/end_use_detail_mapping.yml).
