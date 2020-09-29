clear

cd ../output
import delimited data_misspecified
cd ../code

regress delta x sat wire price, noconst

/* Simple 2SLS method */
ivregress 2sls delta x sat wire (price = w), noconst


/* Nested logit (manually conduct 2SLS) */
quietly {
	regress price x sat wire w x_opp w_opp, noconst
	_predict phat

	regress lnwgs_sat x sat wire w x_opp w_opp, noconst
	_predict lnwgssathat

	regress lnwgs_wire x sat wire w x_opp w_opp, noconst
	_predict lnwgswirehat
}
regress delta x sat wire phat lnwgssathat lnwgswirehat, noconst


/* Nested Logit */

ivregress 2sls delta x sat wire (price lnwgs_sat lnwgs_wire = w x_opp w_opp), noconst 


/* 

/* Somewhat cheating */

regress lnwgs x sat wire w x_opp w_opp, noconst
_predict lnwgshat

gen ls = lnwgshat*sat 
gen lw = lnwgshat*wire

regress delta x sat wire price ls lw, noconst        

*/

