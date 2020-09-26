set more off
clear
log using ../output/analysis.log, replace

import delimited using ../temp/data.csv, clear
rename v1 j
rename v2 t
rename v3 x
rename v4 sat
rename v5 wire
rename v6 p
rename v7 w
rename v8 xi
rename v9 omega
rename v10 s
rename v11 mc

foreach v in x sat wire p w xi omega s mc {
    sum `v'
    matrix row = r(N), r(mean), r(sd), r(min), r(max)
    matrix summary_stats = nullmat(summary_stats) \ row
}
matrix colnames summary_stats = N Mean SD Min Max
assert p > mc
collapse (sum) s, by(t)
assert s <= 1
sum s

outtable using ../output/summary_stats, mat(summary_stats) ///
  format(%9.0fc %9.2fc %9.2fc %9.2fc %9.2fc)

