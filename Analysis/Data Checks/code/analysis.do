set more off
clear
log using ../output/analysis.log, replace

foreach method in fsolve zeta {
    import delimited using ../temp/data_`method'.csv, clear
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
    save ../temp/data_`method'.dta, replace
}

use ../temp/data_fsolve.dta, clear
rename p p_fsolve
rename s s_fsolve
merge 1:1 j t x using ../temp/data_zeta.dta, assert(3) keep(3) ///
    keepusing(p s)
rename p p_zeta
rename s s_zeta
reg p_zeta p_fsolve
corr p_zeta p_fsolve
gen p_diff = p_zeta - p_fsolve
reg s_zeta s_fsolve
corr s_zeta s_fsolve
gen s_diff = s_zeta - s_fsolve
summ *_diff

gen mu_fsolve = p_fsolve - mc
gen mu_zeta = p_zeta - mc
sum mu_*

gen p = p_fsolve
gen s = s_fsolve

foreach v in x sat wire p w xi omega s mc {
    sum `v'
    matrix row = r(N), r(mean), r(sd), r(min), r(max)
    matrix summary_stats = nullmat(summary_stats) \ row
}
matrix rownames summary_stats = x sat wire p w xi omega s mc
matrix colnames summary_stats = N Mean SD Min Max
*assert p > mc
plot p mc
histogram x
graph export ../output/hist_x.pdf, replace
histogram s
graph export ../output/hist_s.pdf, replace
twoway scatter p mc 
graph export ../output/p_mc.pdf, replace
list j t if s == 0
collapse (sum) s, by(t)
assert s <= 1
sum s

outtable using ../output/summary_stats, mat(summary_stats) ///
  format(%9.0fc %9.2fc %9.2fc %9.5fc %9.2fc nobox)

