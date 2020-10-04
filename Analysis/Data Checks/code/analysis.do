set more off
clear
log using ../output/analysis.log, replace

foreach file in fsolve_100 fsolve_200 fsolve_500 fsolve_1000 zeta_1000 {
    import delimited using ../temp/`file'.csv, clear
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
    save ../temp/`file'.dta, replace
}

use ../temp/fsolve_100.dta, clear
rename p p_f100 
rename s s_f100
merge 1:1 j t x using ../temp/fsolve_200.dta, assert(3) keep(3) ///
    keepusing(p s) nogen
rename p p_f200
rename s s_f200
merge 1:1 j t x using ../temp/fsolve_500.dta, assert(3) keep(3) ///
    keepusing(p s) nogen
rename p p_f500
rename s s_f500
merge 1:1 j t x using ../temp/fsolve_1000.dta, assert(3) keep(3) ///
    keepusing(p s) nogen
rename p p_f1000
rename s s_f1000
merge 1:1 j t x using ../temp/zeta_1000.dta, assert(3) keep(3) ///
    keepusing(p s) nogen
rename p p_z1000
rename s s_z1000

foreach v in x sat wire w xi omega mc {
    sum `v'
    matrix row = r(N), r(mean), r(sd), r(min), r(max)
    matrix exog_stats = nullmat(exog_stats) \ row
}
matrix rownames exog_stats = x sat wire w xi omega mc
matrix colnames exog_stats = N Mean SD Min Max
outtable using ../output/exog_stats, mat(exog_stats) ///
  format(%9.0fc %9.2fc %9.2fc %9.5fc %9.2fc) nobox 

foreach suff in f100 f200 f500 f1000 z1000 {
    gen mu_`suff' = p_`suff' - mc
    foreach v in p mu s {
        summ `v'_`suff'
        matrix col = r(mean) \ r(sd) \ r(min) \ r(max)
        matrix endog_stats = nullmat(endog_stats), col
    }
}

matrix bottom = 1 \ 1 \ 1
matrix bottom = bottom, bottom, bottom
foreach v in p mu s {
    corr `v'_f200 `v'_f100
    matrix col = r(rho)  
    gen d`v'= abs(`v'_f200 - `v'_f100)
    summ d`v'
    matrix col = col \ r(mean) \ r(max)
    matrix bottom = bottom, col
    drop d`v'
}
foreach v in p mu s {
    corr `v'_f500 `v'_f200
    matrix col = r(rho)  
    gen d`v'= abs(`v'_f500 - `v'_f200)
    summ d`v'
    matrix col = col \ r(mean) \ r(max)
    matrix bottom = bottom, col
    drop d`v'
}
foreach v in p mu s {
    corr `v'_f1000 `v'_f500
    matrix col = r(rho)  
    gen d`v'= abs(`v'_f1000 - `v'_f500)
    summ d`v'
    matrix col = col \ r(mean) \ r(max)
    matrix bottom = bottom, col
    drop d`v'
}
foreach v in p mu s {
    corr `v'_z1000 `v'_f1000
    matrix col = r(rho)  
    gen d`v'= abs(`v'_z1000 - `v'_f1000)
    summ d`v'
    matrix col = col \ r(mean) \ r(max)
    matrix bottom = bottom, col
    drop d`v'
}

mat li endog_stats
mat li bottom 
matrix endog_stats = endog_stats \ bottom
matrix rownames endog_stats = Mean SD Min Max Correlation MeanDiff MaxDiff 
matrix colnames endog_stats = Price Markup Share Price Markup Share ///
  Price Markup Share Price Markup Share Price Markup Share  
outtable using ../output/endog_stats, mat(endog_stats) format(%9.2fc) nobox

histogram s_z1000
graph export ../output/hist_s.pdf, replace
twoway scatter p_z1000 mc 
graph export ../output/p_mc.pdf, replace


