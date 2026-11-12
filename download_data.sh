#!/bin/sh
# Download the two public data sets used in the study into ./data
mkdir -p data && cd data
for s in cleveland hungarian switzerland va; do
  curl -sO https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.$s.data
done
curl -sL -o cdc_diabetes.csv https://archive.ics.uci.edu/static/public/891/data.csv
