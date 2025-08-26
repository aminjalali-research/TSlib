#!/bin/bash
# Run optimization with proper environment setup

echo "🔧 Setting up environment..."
source /home/amin/anaconda3/bin/activate tslib

echo "🧪 Verifying torch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo "🚀 Starting optimization..."
cd /home/amin/TSlib
python unified/consolidated_metrics_interface.py
