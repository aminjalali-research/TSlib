#!/bin/bash

# Data download helper script for TimesURL

echo "TimesURL Dataset Download Helper"
echo "================================="
echo ""

echo "This script provides guidance for downloading datasets."
echo "Due to licensing restrictions, datasets must be downloaded manually."
echo ""

echo "1. UCR Time Series Classification Archive:"
echo "   - URL: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
echo "   - Download individual datasets or the full archive"
echo "   - Extract to: datasets/UCR/DATASET_NAME/"
echo "   - Each dataset should contain: DATASET_NAME_TRAIN.tsv and DATASET_NAME_TEST.tsv"
echo ""

echo "2. UEA Multivariate Time Series Classification Archive:"
echo "   - URL: http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip"
echo "   - Extract to: datasets/UEA/"
echo "   - Each dataset should be in .arff format or preprocessed .npy format"
echo ""

echo "Popular small UCR datasets for testing:"
echo "- Coffee (56 samples, length 286)"
echo "- ItalyPowerDemand (1096 samples, length 24)" 
echo "- SonyAIBORobotSurface1 (621 samples, length 70)"
echo "- TwoLeadECG (1162 samples, length 82)"
echo ""

echo "Example directory structure after download:"
echo "datasets/"
echo "├── UCR/"
echo "│   ├── Coffee/"
echo "│   │   ├── Coffee_TRAIN.tsv"
echo "│   │   └── Coffee_TEST.tsv"
echo "│   └── ItalyPowerDemand/"
echo "│       ├── ItalyPowerDemand_TRAIN.tsv"
echo "│       └── ItalyPowerDemand_TEST.tsv"
echo "└── UEA/"
echo "    ├── BasicMotions/"
echo "    │   ├── BasicMotions_TRAIN.arff"
echo "    │   └── BasicMotions_TEST.arff"
echo "    └── ..."
echo ""

echo "After downloading datasets, run: bash run_examples.sh"
