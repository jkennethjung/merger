#!/bin/bash
rm -rf ../output/
rm -rf ../temp/

mkdir ../output/
mkdir ../temp/
ln -s ../../../Data/output/data.csv ../temp/data.csv

stata analysis.do 
