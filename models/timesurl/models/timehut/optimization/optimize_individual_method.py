#!/usr/bin/env python3
"""
PyHopper optimization for individual temperature scheduling methods
This script optimizes one method at a time with 10 optimization steps
"""

import subprocess
import json
import time
import sys
import os
from pathlib import Path

def run_training_command(params, method_name, dataset='Chinatown'):
    """Run training command and return accuracy"""
    try:
        # Construct the command with proper format
        cmd = [
            'python', 'train_optimized.py',
            '--loader', 'UCR',
            '--batch-size', '8', 
            '--repr-dims', '320',
            '--epochs', '200',
            '--seed', '2002',
            '--scenario', f'temp_{method_name}',
            '--min-tau', str(params['min_tau']),
            '--max-tau', str(params['max_tau']),
            '--t-max', str(params['t_max']),
            '--dataroot', '/home/amin/TimesURL/datasets',
            dataset,
            f'temp_{method_name}_opt'
        ]
        
        # Add method-specific parameters
        if method_name == 'exponential_decay':
            cmd.extend(['--decay_rate', str(params.get('decay_rate', 0.95))])
        elif method_name == 'step_decay':
            cmd.extend(['--step_size', str(int(params.get('step_size', 8)))])
            cmd.extend(['--gamma', str(params.get('gamma', 0.5))])
        elif method_name == 'polynomial_decay':
            cmd.extend(['--power', str(params.get('power', 2.0))])
        elif method_name == 'sigmoid_decay':
            cmd.extend(['--steepness', str(params.get('steepness', 1.0))])
        elif method_name == 'warmup_cosine':
            cmd.extend(['--warmup_epochs', str(int(params.get('warmup_epochs', 2)))])
        elif method_name == 'cyclic':
            cmd.extend(['--cycle_length', str(params.get('cycle_length', 8.0))])
        
        print(f"Running: {' '.join(cmd)}")
        
        # Change to TimeHUT directory
        original_dir = os.getcwd()
        os.chdir('/home/amin/TimesURL/methods/TimeHUT')
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Change back
        os.chdir(original_dir)
        
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return 0.0
        
        # Parse the result to get accuracy
        result_file = f'/home/amin/TimesURL/methods/TimeHUT/results/UCR_{dataset}_acc_temp_{method_name}_integrated.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                return data['result']['acc']
        else:
            print(f"Result file not found: {result_file}")
            return 0.0
            
    except Exception as e:
        print(f"Error running training: {e}")
        return 0.0

def optimize_linear_decay(steps=10):
    """Optimize linear decay parameters"""
    print(f"\n🔄 Optimizing Linear Decay ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        # Random parameter selection
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'linear')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)  # Brief pause between runs
    
    return {
        'method': 'linear_decay',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def optimize_exponential_decay(steps=10):
    """Optimize exponential decay parameters"""
    print(f"\n🔄 Optimizing Exponential Decay ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0),
            'decay_rate': random.uniform(0.9, 0.99)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'exponential')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)
    
    return {
        'method': 'exponential_decay',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def optimize_step_decay(steps=10):
    """Optimize step decay parameters"""
    print(f"\n🔄 Optimizing Step Decay ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0),
            'step_size': random.randint(3, 15),
            'gamma': random.uniform(0.3, 0.8)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'step')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)
    
    return {
        'method': 'step_decay',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def optimize_sigmoid_decay(steps=10):
    """Optimize sigmoid decay parameters"""
    print(f"\n🔄 Optimizing Sigmoid Decay ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0),
            'steepness': random.uniform(0.5, 3.0)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'sigmoid')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)
    
    return {
        'method': 'sigmoid_decay',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def optimize_polynomial_decay(steps=10):
    """Optimize polynomial decay parameters"""
    print(f"\n🔄 Optimizing Polynomial Decay ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0),
            'power': random.uniform(1.0, 4.0)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'polynomial')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)
    
    return {
        'method': 'polynomial_decay',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def optimize_cyclic(steps=10):
    """Optimize cyclic parameters"""
    print(f"\n🔄 Optimizing Cyclic ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0),
            'cycle_length': random.uniform(2.0, 12.0)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'cyclic')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)
    
    return {
        'method': 'cyclic',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def optimize_warmup_cosine(steps=10):
    """Optimize warmup cosine parameters"""
    print(f"\n🔄 Optimizing Warmup Cosine ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0),
            'warmup_epochs': random.randint(1, 8)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'warmup_cosine')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)
    
    return {
        'method': 'warmup_cosine',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def optimize_cosine_annealing(steps=10):
    """Optimize cosine annealing parameters"""
    print(f"\n🔄 Optimizing Cosine Annealing ({steps} steps)")
    
    best_accuracy = 0.0
    best_params = None
    
    for step in range(steps):
        import random
        params = {
            'min_tau': random.uniform(0.05, 0.2),
            'max_tau': random.uniform(0.6, 0.9),
            't_max': random.uniform(15.0, 35.0)
        }
        
        print(f"Step {step+1}/{steps}: Testing params {params}")
        accuracy = run_training_command(params, 'cosine')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"✅ New best: {accuracy:.4f}")
        
        time.sleep(1)
    
    return {
        'method': 'cosine_annealing',
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'optimization_steps': steps
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_individual_method.py <method_name>")
        print("Available methods: linear, exponential, step, sigmoid")
        return
    
    method = sys.argv[1].lower()
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"🚀 Starting individual optimization for {method} method")
    print(f"📊 Dataset: Chinatown")
    print(f"🔄 Optimization steps: {steps}")
    print("="*50)
    
    start_time = time.time()
    
    if method == 'linear':
        result = optimize_linear_decay(steps)
    elif method == 'exponential':
        result = optimize_exponential_decay(steps)
    elif method == 'step':
        result = optimize_step_decay(steps)
    elif method == 'sigmoid':
        result = optimize_sigmoid_decay(steps)
    elif method == 'polynomial':
        result = optimize_polynomial_decay(steps)
    elif method == 'cyclic':
        result = optimize_cyclic(steps)
    elif method == 'warmup_cosine':
        result = optimize_warmup_cosine(steps)
    elif method == 'cosine':
        result = optimize_cosine_annealing(steps)
    else:
        print(f"❌ Unknown method: {method}")
        return
    
    optimization_time = time.time() - start_time
    result['optimization_time'] = optimization_time
    
    # Save results
    results_file = f'/home/amin/TimesURL/individual_optimization_{method}.json'
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n🏁 Optimization complete!")
    print(f"⏱️  Total time: {optimization_time:.2f} seconds")
    print(f"🎯 Best accuracy: {result['best_accuracy']:.4f}")
    print(f"⚙️  Best parameters: {result['best_params']}")
    print(f"💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
