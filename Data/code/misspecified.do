clear all


cd ../output
import delimited data_misspecified
cd ../code

/* (1) OLS */

eststo: regress delta x sat wire price, noconst 


/* (2) Simple 2SLS method */
eststo: ivregress 2sls delta x sat wire (price = w), noconst

/* (3) Nested Logit */


/* (3-i) Nested Logit (assuming correlation coeffs are identical) */

eststo: ivregress 2sls delta x sat wire (price lnwgs = w x_opp w_opp), noconst

/*
qui{
	regress price x sat wire w x_opp w_opp, noconst
	_predict phat

	regress lnwgs x sat wire w x_opp w_opp, noconst
	_predict lnwgshat
}
regress delta x sat wire phat lnwgshat, noconst        

*/


/* (3-ii)Nested Logit (using ivreg command) */

eststo: ivregress 2sls delta x sat wire (price lnwgs_sat lnwgs_wire = w x_opp w_opp), noconst first


/* (3-ii') Nested logit (manually conduct 2SLS) */
quietly {
	regress price x sat wire w x_opp w_opp, noconst
	_predict phat_
	
	regress lnwgs_sat x sat wire w x_opp w_opp, noconst
	_predict lnwgssathat

	regress lnwgs_wire x sat wire w x_opp w_opp, noconst
	_predict lnwgswirehat
}
/* eststo: regress delta x sat wire phat_ lnwgssathat lnwgswirehat, noconst */
 

 
/*  output tex code for the table of estimated coefficients

esttab using ../output/misspecified.tex, label title(Regression table\label{tab1}) replace

*/


