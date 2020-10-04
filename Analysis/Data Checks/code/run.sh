#!/bin/bash
rm -rf ../output/
rm -rf ../temp/

mkdir ../output/
mkdir ../temp/
ln -s ../../../Data/output/fsolve_100.csv ../temp/fsolve_100.csv
ln -s ../../../Data/output/fsolve_200.csv ../temp/fsolve_200.csv
ln -s ../../../Data/output/fsolve_500.csv ../temp/fsolve_500.csv
ln -s ../../../Data/output/fsolve_1000.csv ../temp/fsolve_1000.csv
ln -s ../../../Data/output/zeta_1000.csv ../temp/zeta_1000.csv

stata analysis.do 
