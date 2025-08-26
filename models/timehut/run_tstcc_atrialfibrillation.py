#!/usr/bin/env python3
"""
TS-TCC AtrialFibrillation Runner Script
======================================

This script runs the TS-TCC baseline model on the AtrialFibrillation dataset
with proper configuration and data handling.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tstcc_atrialfibrillation(epochs=80, batch_size=16, seed=42):
    """Run TS-TCC on AtrialFibrillation dataset"""
    
    # Set paths
    tstcc_dir = Path("/home/amin/TSlib/models/ts_contrastive/tsm/baselines/TS-TCC")
    
    if not tstcc_dir.exists():
        print(f"❌ TS-TCC directory not found: {tstcc_dir}")
        return False
    
    # Change to TS-TCC directory
    os.chdir(tstcc_dir)
    
    # Run TS-TCC with AtrialFibrillation configuration
    cmd = [
        'python', 'main.py',
        '--selected_dataset', 'AtrialFibrillation',
        '--training_mode', 'self_supervised',
        '--seed', str(seed),
        '--logs_save_dir', 'experiments_logs',
        '--experiment_description', 'atrialfibrillation_test',
        '--run_description', 'enhanced_run'
    ]
    
    print(f"🚀 Running TS-TCC on AtrialFibrillation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print("✅ TS-TCC completed successfully")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.TimeoutExpired:
        print("❌ TS-TCC timed out (30 minutes)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ TS-TCC failed with return code {e.returncode}")
        print("STDERR:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"❌ TS-TCC failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run TS-TCC on AtrialFibrillation')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    success = run_tstcc_atrialfibrillation(args.epochs, args.batch_size, args.seed)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
