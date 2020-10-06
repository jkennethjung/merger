#!/bin/bash
rm -rf ../output/
rm -rf ../temp/

mkdir ../output/
mkdir ../temp/
ln -s ../../../Data/output/* ../temp/

stata analysis.do 
