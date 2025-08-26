#!/bin/bash

# GPU-enabled run script for TimesURL with your datasets

echo "=== TimesURL GPU Training Script ==="

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "timesurl" ]]; then
    echo "Please activate the timesurl environment first:"
    echo "conda activate timesurl"
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Function to run TimesURL training
run_timesurl() {
    local dataset=$1
    local loader=$2
    local experiment_name=$3
    local gpu_id=${4:-0}
    
    echo "Training TimesURL on $dataset dataset..."
    echo "Using GPU: $gpu_id"
    
    python train.py $dataset $experiment_name \
        --loader $loader \
        --gpu $gpu_id \
        --batch-size 16 \
        --lr 0.0001 \
        --repr-dims 320 \
        --max-train-length 3000 \
        --epochs 100 \
        --eval \
        --temp 1.0 \
        --lmd 0.01 \
        --seed 42
}

# Quick test on a small UCR dataset
echo "=== Quick Test: Coffee Dataset ==="
if [ -d "datasets/UCR/Coffee" ]; then
    run_timesurl Coffee UCR coffee_gpu_test 0
else
    echo "Coffee dataset not found in datasets/UCR/Coffee/"
fi

echo ""

# Run on another UCR dataset
echo "=== UCR Dataset: ECG200 ==="
if [ -d "datasets/UCR/ECG200" ]; then
    run_timesurl ECG200 UCR ecg200_gpu_test 0
else
    echo "ECG200 dataset not found in datasets/UCR/ECG200/"
fi

echo ""

# Run on UEA dataset
echo "=== UEA Dataset: BasicMotions ==="
if [ -d "datasets/UEA/BasicMotions" ]; then
    run_timesurl BasicMotions UEA basicmotions_gpu_test 0
else
    echo "BasicMotions dataset not found in datasets/UEA/BasicMotions/"
fi

echo ""
echo "=== Available Datasets ==="
echo "UCR datasets:"
if [ -d "datasets/UCR" ]; then
    ls datasets/UCR/ | head -10
    echo "... and more"
else
    echo "No UCR directory found"
fi

echo ""
echo "UEA datasets:"
if [ -d "datasets/UEA" ]; then
    ls datasets/UEA/ | head -10
    echo "... and more"
else
    echo "No UEA directory found"
fi

echo ""
echo "=== Results Location ==="
echo "Training results are saved in: ./training/"
echo "Model files: ./training/[dataset]__[experiment_name]/model.pkl"
echo "Evaluation results: ./training/[dataset]__[experiment_name]/eval_res.pkl"

echo ""
echo "=== Custom Run Example ==="
echo "To run on any dataset:"
echo "python train.py DATASET_NAME experiment_name --loader UCR --gpu 0 --eval"
echo "python train.py DATASET_NAME experiment_name --loader UEA --gpu 0 --eval"
