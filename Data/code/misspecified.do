clear

use ../output/misspecified.dta


regress delta x sat wire price, noconst

ivregress 2sls delta x sat wire (price = w), noconst




ivregress 2sls delta x sat wire (lnwgssat lnwgswire price = w x_opp w_opp), noconst

regress price x sat wire w x_opp w_opp
_predict phat

regress lnwgssat x sat wire w x_opp w_opp, noconst
_predict lnwgssathat

regress lnwgswire x sat wire w x_opp w_opp, noconst
_predict lnwgswirehat

regress delta x sat wire price lnwgssathat lnwgswirehat, noconst

regress lnwgs x sat wire w x_opp w_opp, noconst
_predict lnwgshat

gen ls = lnwgshat*sat 
gen lw = lnwgshat*wire

regress delta x sat wire price ls lw, noconst

