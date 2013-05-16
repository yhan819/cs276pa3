#! /usr/bin/zsh 
  rm -f output1.txt
  sh rank.sh 1 queryDocTrainData $1 $2 $3 $4 $5 > output1.txt 
  python ndcg.py output1.txt queryDocTrainRel
