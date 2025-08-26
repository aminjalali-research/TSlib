"""
Pyhopper Optimization for Enhanced Temperature Schedulers
Automatically optimize hyperparameters for all enhanced cosine annealing variants
"""

import os
import sys
import json
import time
import pyhopper
import subprocess
from pathlib import Path

# Add current directory to path for imports
sys.path.append('/home/amin/TimesURL/methods/TimeHUT')

def run_training_command(params, method_name, dataset='Chinatown'):
    """Run training with given parameters and return accuracy"""
    
    # Base command
    cmd = [
        'python', 'train_optimized.py',
        dataset, f'{method_name}_optimized',
        '--loader', 'UCR',
        '--gpu', '0',
        '--batch-size', '8',
        '--epochs', '200',
        '--eval',
        '--seed', '2002',
        '--method', 'acc',
        '--dataroot', '/home/amin/TimesURL/datasets'
    ]
    
    # Add scheduler-specific parameters
    if method_name == 'cosine_annealing':
        cmd.extend([
            '--scenario', 'temp_cosine',
            '--min-tau', str(params['min_tau']),
            '--max-tau', str(params['max_tau']),
            '--t-max', str(params['t_max']),
            '--phase', str(params['phase']),
            '--frequency', str(params['frequency']),
            '--bias', str(params['bias'])
        ])
    elif method_name == 'multi_cycle_cosine':
        cmd.extend([
            '--scenario', 'temp_multi_cycle_cosine',
            '--min-tau', str(params['min_tau']),
            '--max-tau', str(params['max_tau']),
            '--t-max', str(params['t_max']),
            '--num-cycles', str(int(params['num_cycles'])),
            '--decay-factor', str(params['decay_factor'])
        ])
    elif method_name == 'adaptive_cosine_annealing':
        cmd.extend([
            '--scenario', 'temp_adaptive_cosine',
            '--min-tau', str(params['min_tau']),
            '--max-tau', str(params['max_tau']),
            '--t-max', str(params['t_max']),
            '--momentum', str(params['momentum']),
            '--adaptation-rate', str(params['adaptation_rate'])
        ])
    elif method_name == 'cosine_with_restarts':
        cmd.extend([
            '--scenario', 'temp_cosine_restarts',
            '--min-tau', str(params['min_tau']),
            '--max-tau', str(params['max_tau']),
            '--t-max', str(params['t_max']),
            '--restart-period', str(params['restart_period']),
            '--restart-mult', str(params['restart_mult'])
        ])
    
    try:
        # Change to the correct directory
        result = subprocess.run(
            cmd, 
            cwd='/home/amin/TimesURL/methods/TimeHUT',
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return 0.0
        
        # Parse accuracy from output
        lines = result.stdout.split('\n')
        for line in reversed(lines):
            if 'Accuracy:' in line:
                try:
                    acc_str = line.split('Accuracy:')[1].strip()
                    accuracy = float(acc_str)
                    return accuracy
                except (ValueError, IndexError):
                    continue
        
        print("Could not find accuracy in output")
        return 0.0
        
    except subprocess.TimeoutExpired:
        print("Training timed out")
        return 0.0
    except Exception as e:
        print(f"Error running training: {e}")
        return 0.0

def optimize_cosine_annealing(steps=20):
    """Optimize enhanced cosine annealing parameters"""
    print("🔍 Optimizing Enhanced Cosine Annealing...")
    
    search_space = pyhopper.Search({
        "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
        "max_tau": pyhopper.float(0.6, 0.9, "0.02f"), 
        "t_max": pyhopper.float(15, 35, "1.0f"),
        "phase": pyhopper.float(0.0, 6.28, "0.1f"),  # 0 to 2π
        "frequency": pyhopper.float(0.8, 2.0, "0.1f"),
        "bias": pyhopper.float(-0.05, 0.05, "0.01f")
    })
    
    def objective(params):
        return run_training_command(params, 'cosine_annealing')
    
    best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
    
    return {
        'method': 'cosine_annealing',
        'best_params': best_params,
        'best_accuracy': objective(best_params)
    }

def optimize_multi_cycle_cosine(steps=25):
    """Optimize multi-cycle cosine annealing parameters"""
    print("🔍 Optimizing Multi-Cycle Cosine Annealing...")
    
    search_space = pyhopper.Search({
        "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
        "max_tau": pyhopper.float(0.6, 0.9, "0.02f"),
        "t_max": pyhopper.float(15, 35, "1.0f"), 
        "num_cycles": pyhopper.int(2, 8),  # 2 to 8 cycles
        "decay_factor": pyhopper.float(0.6, 0.95, "0.05f")
    })
    
    def objective(params):
        return run_training_command(params, 'multi_cycle_cosine')
    
    best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
    
    return {
        'method': 'multi_cycle_cosine', 
        'best_params': best_params,
        'best_accuracy': objective(best_params)
    }

def optimize_adaptive_cosine(steps=20):
    """Optimize adaptive cosine annealing parameters"""
    print("🔍 Optimizing Adaptive Cosine Annealing...")
    
    search_space = pyhopper.Search({
        "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
        "max_tau": pyhopper.float(0.6, 0.9, "0.02f"),
        "t_max": pyhopper.float(15, 35, "1.0f"),
        "momentum": pyhopper.float(0.8, 0.99, "0.01f"),
        "adaptation_rate": pyhopper.float(0.05, 0.3, "0.01f")
    })
    
    def objective(params):
        return run_training_command(params, 'adaptive_cosine_annealing')
    
    best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
    
    return {
        'method': 'adaptive_cosine_annealing',
        'best_params': best_params,
        'best_accuracy': objective(best_params)
    }

def optimize_cosine_restarts(steps=20):
    """Optimize cosine with restarts parameters"""
    print("🔍 Optimizing Cosine with Restarts...")
    
    search_space = pyhopper.Search({
        "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
        "max_tau": pyhopper.float(0.6, 0.9, "0.02f"),
        "t_max": pyhopper.float(15, 35, "1.0f"),
        "restart_period": pyhopper.float(5, 15, "0.5f"),
        "restart_mult": pyhopper.float(1.1, 2.0, "0.1f")
    })
    
    def objective(params):
        return run_training_command(params, 'cosine_with_restarts')
    
    best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
    
    return {
        'method': 'cosine_with_restarts',
        'best_params': best_params,
        'best_accuracy': objective(best_params)
    }

def run_comprehensive_optimization():
    """Run comprehensive optimization for all enhanced schedulers"""
    print("🚀 Starting Comprehensive Scheduler Optimization...")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    # Optimize each method
    methods_to_optimize = [
        ('multi_cycle_cosine', optimize_multi_cycle_cosine, 25),
        ('cosine_annealing', optimize_cosine_annealing, 20),
        ('adaptive_cosine_annealing', optimize_adaptive_cosine, 20),
        ('cosine_with_restarts', optimize_cosine_restarts, 20)
    ]
    
    for method_name, optimizer_func, steps in methods_to_optimize:
        print(f"\n📊 Optimizing {method_name}...")
        method_start = time.time()
        
        try:
            result = optimizer_func(steps)
            result['optimization_time'] = time.time() - method_start
            results.append(result)
            
            print(f"✅ {method_name} optimization completed!")
            print(f"   Best Accuracy: {result['best_accuracy']:.4f}")
            print(f"   Time: {result['optimization_time']:.1f}s")
            
        except Exception as e:
            print(f"❌ Error optimizing {method_name}: {e}")
            results.append({
                'method': method_name,
                'error': str(e),
                'optimization_time': time.time() - method_start
            })
    
    # Save results
    total_time = time.time() - start_time
    optimization_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_optimization_time': total_time,
        'dataset': 'Chinatown',
        'results': results
    }
    
    output_file = '/home/amin/TimesURL/methods/TimeHUT/pyhopper_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if 'best_accuracy' in r]
    if successful_results:
        best_method = max(successful_results, key=lambda x: x['best_accuracy'])
        
        print(f"🥇 Best Method: {best_method['method']}")
        print(f"🎯 Best Accuracy: {best_method['best_accuracy']:.4f}")
        print(f"⚡ Best Parameters:")
        for param, value in best_method['best_params'].items():
            if isinstance(value, float):
                print(f"   {param}: {value:.3f}")
            else:
                print(f"   {param}: {value}")
        
        print(f"\n📊 All Results:")
        for result in successful_results:
            print(f"   {result['method']}: {result['best_accuracy']:.4f}")
    
    print(f"\n⏱️ Total Time: {total_time:.1f}s")
    print(f"💾 Results saved to: {output_file}")
    
    return optimization_results

if __name__ == "__main__":
    # Run comprehensive optimization
    results = run_comprehensive_optimization()
