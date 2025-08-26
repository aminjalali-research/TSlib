#!/usr/bin/env python3
"""
TFC AtrialFibrillation Runner Script
===================================

This script runs the TFC baseline model on the AtrialFibrillation dataset
with proper configuration and data handling.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tfc_atrialfibrillation(epochs=40, batch_size=16, seed=42):
    """Run TFC on AtrialFibrillation dataset"""
    
    # Set paths
    tfc_dir = Path("/home/amin/TSlib/models/ts_contrastive/tsm/baselines/TFC")
    
    if not tfc_dir.exists():
        print(f"❌ TFC directory not found: {tfc_dir}")
        return False
    
    # Change to TFC directory
    os.chdir(tfc_dir)
    
    # Run TFC with AtrialFibrillation configuration
    cmd = [
        'python', 'main.py',
        '--pretrain_dataset', 'AtrialFibrillation',
        '--target_dataset', 'AtrialFibrillation', 
        '--training_mode', 'fine_tune_test',
        '--seed', str(seed),
        '--logs_save_dir', 'experiments_logs'
    ]
    
    print(f"🚀 Running TFC on AtrialFibrillation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print("✅ TFC completed successfully")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.TimeoutExpired:
        print("❌ TFC timed out (30 minutes)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ TFC failed with return code {e.returncode}")
        print("STDERR:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"❌ TFC failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run TFC on AtrialFibrillation')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    success = run_tfc_atrialfibrillation(args.epochs, args.batch_size, args.seed)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
