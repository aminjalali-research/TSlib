#!/usr/bin/env python3
"""
Mixing-up AtrialFibrillation Runner Script
==========================================

This script runs the Mixing-up baseline model on the AtrialFibrillation dataset
with proper configuration and data handling.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_mixingup_atrialfibrillation(epochs=50, batch_size=16, seed=42):
    """Run Mixing-up on AtrialFibrillation dataset"""
    
    # Set paths
    mixingup_dir = Path("/home/amin/TSlib/models/ts_contrastive/tsm/baselines/Mixing-up")
    
    if not mixingup_dir.exists():
        print(f"❌ Mixing-up directory not found: {mixingup_dir}")
        return False
    
    # Check if data exists
    data_dir = mixingup_dir / "data" / "AtrialFibrillation"
    if not data_dir.exists():
        print(f"❌ AtrialFibrillation data not found in {data_dir}")
        return False
    
    # Change to Mixing-up directory
    os.chdir(mixingup_dir)
    
    # Run Mixing-up with AtrialFibrillation data
    cmd = [
        'python', 'train_model.py',  # Assuming this is the main training script
        '--dataset', 'AtrialFibrillation',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--seed', str(seed)
    ]
    
    print(f"🚀 Running Mixing-up on AtrialFibrillation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print("✅ Mixing-up completed successfully")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.TimeoutExpired:
        print("❌ Mixing-up timed out (30 minutes)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Mixing-up failed with return code {e.returncode}")
        print("STDERR:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"❌ Mixing-up failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Mixing-up on AtrialFibrillation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    success = run_mixingup_atrialfibrillation(args.epochs, args.batch_size, args.seed)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
