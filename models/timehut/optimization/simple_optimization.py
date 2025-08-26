#!/usr/bin/env python3
"""
Simplified PyHopper Optimization for Key Temperature Scheduling Methods
Test with a few key methods first, then expand if successful
"""

import os
import sys
import json
import time
import subprocess
import random
from pathlib import Path

# Add the TimeHUT directory to path
timehut_dir = '/home/amin/TimesURL/methods/TimeHUT'
if timehut_dir not in sys.path:
    sys.path.append(timehut_dir)

def run_training_command(params, method_name):
    """Run training command with given parameters and return accuracy"""
    
    # Map method names to scenario names
    scenario_mapping = {
        'cosine_annealing': 'temp_cosine',
        'linear_decay': 'temp_linear', 
        'exponential_decay': 'temp_exponential',
        'sigmoid_decay': 'temp_sigmoid',
        'cyclic': 'temp_cyclic',
    }
    
    scenario = scenario_mapping.get(method_name, 'temp_cosine')
    
    # Build command with method-specific parameters
    cmd = [
        'python', 'train_optimized.py',
        '--loader', 'UCR',
        '--batch-size', str(params.get('batch_size', 8)),
        '--epochs', str(params.get('epochs', 200)),
        '--seed', str(params.get('seed', 2002)),
        '--scenario', scenario,
        '--min-tau', str(params['min_tau']),
        '--max-tau', str(params['max_tau']),
        '--t-max', str(params['t_max']),
        'Chinatown', 'temp_optimization_test'
    ]
    
    # Add method-specific parameters
    if method_name == 'cosine_annealing':
        cmd.extend(['--phase', str(params.get('phase', 0.0))])
        cmd.extend(['--frequency', str(params.get('frequency', 1.0))])
        cmd.extend(['--bias', str(params.get('bias', 0.0))])
    elif method_name == 'exponential_decay':
        cmd.extend(['--decay_rate', str(params.get('decay_rate', 0.95))])
    elif method_name == 'sigmoid_decay':
        cmd.extend(['--steepness', str(params.get('steepness', 1.0))])
    elif method_name == 'cyclic':
        cmd.extend(['--cycle_length', str(params.get('cycle_length', 8.3))])
    
    try:
        # Change to TimeHUT directory and run command
        result = subprocess.run(cmd, cwd=timehut_dir, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"    Training failed for {method_name}: {result.stderr}")
            return 0.0
        
        # Parse accuracy from output
        output = result.stdout
        if "Test accuracy:" in output:
            for line in output.split('\n'):
                if "Test accuracy:" in line:
                    acc_str = line.split("Test accuracy:")[1].strip()
                    accuracy = float(acc_str)
                    return accuracy
        
        # Fallback: try to read from results file
        result_file = f'/home/amin/TimesURL/methods/TimeHUT/results/UCR_Chinatown_acc_{scenario}_integrated.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                return data.get('result', {}).get('acc', 0.0)
        
        return 0.0
        
    except subprocess.TimeoutExpired:
        print(f"    Training timeout for {method_name}")
        return 0.0
    except Exception as e:
        print(f"    Error running training for {method_name}: {e}")
        return 0.0

def random_search_optimization(method_name, param_ranges, steps=10):
    """Simple random search optimization"""
    
    best_params = None
    best_accuracy = 0.0
    
    print(f"    Starting {steps} random search steps for {method_name}...")
    
    for step in range(steps):
        # Generate random parameters
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int):
                params[param_name] = random.randint(min_val, max_val)
            else:
                params[param_name] = random.uniform(min_val, max_val)
        
        # Run training
        accuracy = run_training_command(params, method_name)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params.copy()
        
        print(f"    Step {step+1}/{steps}: accuracy = {accuracy:.4f} (best: {best_accuracy:.4f})")
    
    return best_params, best_accuracy

def optimize_method(method_name, param_ranges, steps=10):
    """Optimize a single method using random search"""
    
    print(f"\n🔧 Optimizing {method_name} ({steps} steps)...")
    method_start_time = time.time()
    
    try:
        best_params, best_accuracy = random_search_optimization(method_name, param_ranges, steps)
        method_time = time.time() - method_start_time
        
        result = {
            'method': method_name,
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'optimization_time': method_time,
            'steps_completed': steps
        }
        
        print(f"✅ {method_name}: Best accuracy = {best_accuracy:.4f}")
        print(f"   Parameters: {best_params}")
        print(f"   Time: {method_time:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"❌ Failed to optimize {method_name}: {e}")
        return {
            'method': method_name,
            'best_params': {},
            'best_accuracy': 0.0,
            'optimization_time': 0.0,
            'error': str(e)
        }

def main():
    print("🚀 Starting Random Search Optimization for Key Temperature Schedulers")
    print("="*70)
    
    # Define key methods with their parameter ranges
    methods_to_optimize = {
        'linear_decay': {
            'min_tau': (0.05, 0.2),
            'max_tau': (0.6, 0.9),
            't_max': (15.0, 35.0)
        },
        'exponential_decay': {
            'min_tau': (0.05, 0.2),
            'max_tau': (0.6, 0.9),
            't_max': (15.0, 35.0),
            'decay_rate': (0.9, 0.99)
        },
        'sigmoid_decay': {
            'min_tau': (0.05, 0.2),
            'max_tau': (0.6, 0.9),
            't_max': (15.0, 35.0),
            'steepness': (0.5, 2.0)
        },
        'cyclic': {
            'min_tau': (0.05, 0.2),
            'max_tau': (0.6, 0.9),
            't_max': (15.0, 35.0),
            'cycle_length': (2.0, 15.0)
        },
        'cosine_annealing': {
            'min_tau': (0.05, 0.2),
            'max_tau': (0.6, 0.9),
            't_max': (15.0, 35.0),
            'phase': (0.0, 6.3),
            'frequency': (0.5, 2.0),
            'bias': (-0.1, 0.1)
        }
    }
    
    all_results = []
    total_start_time = time.time()
    
    for method_name, param_ranges in methods_to_optimize.items():
        result = optimize_method(method_name, param_ranges, steps=10)
        all_results.append(result)
    
    total_time = time.time() - total_start_time
    
    # Save results
    results_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_optimization_time': total_time,
        'dataset': 'Chinatown',
        'optimization_steps_per_method': 10,
        'optimization_method': 'random_search',
        'total_methods': len(methods_to_optimize),
        'results': all_results
    }
    
    results_file = '/home/amin/TimesURL/methods/TimeHUT/random_search_optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print(f"\n🏆 OPTIMIZATION SUMMARY")
    print("="*40)
    print(f"Total methods optimized: {len(all_results)}")
    print(f"Total optimization time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per method: {total_time/len(all_results):.1f} seconds")
    print(f"Results saved to: {results_file}")
    
    # Sort by best accuracy
    successful_results = [r for r in all_results if 'error' not in r and r['best_accuracy'] > 0]
    successful_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    print(f"\n📊 TOP PERFORMERS:")
    for i, result in enumerate(successful_results):
        print(f"{i+1}. {result['method']}: {result['best_accuracy']:.4f} accuracy")
        print(f"   Best params: {result['best_params']}")
    
    print(f"\n✅ Random search optimization finished!")
    print(f"📁 Results file: {results_file}")

if __name__ == "__main__":
    main()
