#!/bin/bash

# Quick test script for small UCR datasets

echo "Quick test with minimal settings..."

# Test with a small dataset if available
DATASET="Coffee"  # Change this to any available dataset
EXPERIMENT_NAME="quick_test"

if [ -d "datasets/UCR/$DATASET" ]; then
    echo "Running quick test on $DATASET dataset..."
    
    python train.py $DATASET $EXPERIMENT_NAME \
        --loader UCR \
        --gpu 0 \
        --batch-size 4 \
        --lr 0.001 \
        --repr-dims 128 \
        --epochs 5 \
        --eval \
        --temp 1.0 \
        --lmd 0.01
        
    echo "Quick test completed! Check training/${DATASET}__${EXPERIMENT_NAME}_* for results"
else
    echo "Dataset $DATASET not found in datasets/UCR/"
    echo "Please download UCR datasets or modify the DATASET variable in this script"
    echo ""
    echo "Available datasets in datasets/UCR/:"
    ls datasets/UCR/ 2>/dev/null || echo "No datasets found"
fi
