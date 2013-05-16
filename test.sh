#!/bin/bash
# My first script

rm -f output2.txt
./rank.sh 2 queryDocTrainData > output2.txt
python ndcg.py output2.txt queryDocTrainRel

