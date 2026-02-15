# use_share_j_frac.csv

End-use allocation shares used to split total demand/consumption into end-use buckets.
- Columns: t,m,j,value
- Unit: 0..1
- Constraint: sum over j must equal 1 for each (t,m).
- Optional if your SD demand module is already end-use specific.
