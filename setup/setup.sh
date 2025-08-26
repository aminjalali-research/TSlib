#!/bin/bash

# Setup script for TSlib project

echo "Setting up TSlib environment..."

# Create conda environment
echo "Creating conda environment 'tslib'..."
conda env create -f environment.yml

# Check if environment was created successfully
if [ $? -eq 0 ]; then
    echo "✓ Conda environment 'tslib' created successfully!"
    echo ""
    echo "Activating environment to test installation..."
    
    # Activate environment and test
    eval "$(conda shell.bash hook)"
    conda activate tslib
    
    # Check GPU availability
    echo "Checking GPU availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    
    echo "✓ Dependencies installed successfully!"
else
    echo "✗ Failed to create conda environment. Please check for errors above."
    exit 1
fi

# Create results directory 
echo "Creating results directory..."
mkdir -p results

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate tslib"
echo "2. Run comprehensive analysis: python unified/consolidated_metrics_interface.py"
echo "3. Or run quick tests: bash setup/quick_test.sh"
echo ""
echo "Your datasets are available at datasets/"
