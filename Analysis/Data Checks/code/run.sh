#!/bin/bash
rm -rf ../output/
rm -rf ../temp/

mkdir ../output/
mkdir ../temp/
ln -s ../../../Data/output/data_fsolve.csv ../temp/data_fsolve.csv
ln -s ../../../Data/output/data_zeta.csv ../temp/data_zeta.csv

stata analysis.do 
