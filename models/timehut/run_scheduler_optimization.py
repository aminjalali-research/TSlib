#!/usr/bin/env python3
"""
Enhanced Temperature Scheduler Optimization Script
================================================

This script uses the integrated optimization functionality from temperature_schedulers.py
to run comprehensive optimization and benchmarking of all temperature scheduling methods.

Features:
✅ PyHopper optimization for multiple schedulers
✅ Systematic benchmark comparison
✅ Statistical analysis with multiple trials
✅ Comprehensive results reporting
✅ Integration with train_unified_comprehensive.py

Usage Examples:
  # Run comprehensive optimization
  python run_scheduler_optimization.py --mode optimize --dataset Chinatown
  
  # Run benchmark comparison
  python run_scheduler_optimization.py --mode benchmark --dataset Chinatown --trials 5
  
  # Run both optimization and benchmark
  python run_scheduler_optimization.py --mode both --dataset Chinatown

Author: TimeHUT Enhanced Framework
Date: August 27, 2025
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from temperature_schedulers import SchedulerOptimizer, SchedulerBenchmark, PYHOPPER_AVAILABLE

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Enhanced Temperature Scheduler Optimization and Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Modes:
  optimize    : Run PyHopper optimization for multiple schedulers
  benchmark   : Run systematic benchmark comparison of all schedulers
  both        : Run both optimization and benchmarking
  
Supported Schedulers:
  - cosine_annealing: Enhanced cosine annealing with phase/frequency control
  - multi_cycle_cosine: Multi-cycle cosine with amplitude decay
  - step_decay: Step-wise decay with configurable parameters
  - exponential_decay: Exponential decay with tunable rate
  - polynomial_decay: Polynomial decay with power control
  - sigmoid_decay: Sigmoid-based smooth decay
  - warmup_cosine: Linear warmup + cosine annealing
  - constant: Constant temperature baseline
  - cyclic: Cyclic sawtooth pattern
  - cosine_with_restarts: SGDR-style cosine with restarts
  - adaptive_cosine_annealing: Performance-aware adaptive cosine
  - linear_decay: Simple linear decay
  - no_scheduling: No temperature control (baseline)

Examples:
  # Quick optimization test
  python run_scheduler_optimization.py --mode optimize --dataset Chinatown --steps 10
  
  # Comprehensive benchmark
  python run_scheduler_optimization.py --mode benchmark --dataset Chinatown --trials 5 --epochs 200
  
  # Full analysis
  python run_scheduler_optimization.py --mode both --dataset Chinatown --steps 20 --trials 3
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, default='Chinatown',
                       help='Dataset name to use for optimization/benchmarking')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['optimize', 'benchmark', 'both'],
                       help='Operation mode')
    
    # Optimization parameters
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of PyHopper optimization steps per scheduler')
    parser.add_argument('--methods', type=str, nargs='*', default=None,
                       help='Specific schedulers to optimize (default: all supported)')
    
    # Benchmark parameters
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials per scheduler for benchmarking')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs for benchmarking')
    parser.add_argument('--benchmark-schedulers', type=str, nargs='*', default=None,
                       help='Specific schedulers to benchmark (default: all available)')
    
    # System parameters
    parser.add_argument('--base-dir', type=str, default='/home/amin/TSlib/models/timehut',
                       help='Base directory for TimeHUT framework')
    parser.add_argument('--output-dir', type=str, default='optimization_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 Enhanced Temperature Scheduler Analysis")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔧 Mode: {args.mode}")
    print(f"📁 Base Directory: {args.base_dir}")
    print("=" * 60)
    
    # Check PyHopper availability for optimization
    if args.mode in ['optimize', 'both'] and not PYHOPPER_AVAILABLE:
        print("❌ Error: PyHopper is required for optimization mode but not available")
        print("Please install PyHopper: pip install pyhopper")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    results = {}
    start_time = time.time()
    
    # =====================================================
    # OPTIMIZATION MODE
    # =====================================================
    if args.mode in ['optimize', 'both']:
        print("\n🔍 Starting PyHopper Optimization...")
        print("-" * 40)
        
        optimizer = SchedulerOptimizer(base_dir=args.base_dir, dataset=args.dataset)
        
        # Define methods to optimize
        if args.methods:
            # User-specified methods
            optimization_methods = []
            method_mapping = {
                'cosine_annealing': (optimizer.optimize_cosine_annealing, args.steps),
                'multi_cycle_cosine': (optimizer.optimize_multi_cycle_cosine, args.steps),
                'step_decay': (optimizer.optimize_step_decay, args.steps),
                'exponential_decay': (optimizer.optimize_exponential_decay, args.steps)
            }
            
            for method in args.methods:
                if method in method_mapping:
                    optimization_methods.append((method, method_mapping[method][0], method_mapping[method][1]))
                else:
                    print(f"⚠️ Warning: Unknown optimization method '{method}' - skipping")
        else:
            # Default methods
            optimization_methods = [
                ('cosine_annealing', optimizer.optimize_cosine_annealing, args.steps),
                ('multi_cycle_cosine', optimizer.optimize_multi_cycle_cosine, args.steps),
                ('step_decay', optimizer.optimize_step_decay, args.steps),
                ('exponential_decay', optimizer.optimize_exponential_decay, args.steps)
            ]
        
        # Run optimization
        try:
            optimization_results = optimizer.run_comprehensive_optimization(optimization_methods)
            results['optimization'] = optimization_results
            
            # Save optimization results
            opt_output_file = output_dir / f"optimization_{args.dataset}_{int(time.time())}.json"
            with open(opt_output_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            
            print(f"💾 Optimization results saved to: {opt_output_file}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            results['optimization'] = {'error': str(e)}
    
    # =====================================================
    # BENCHMARK MODE
    # =====================================================
    if args.mode in ['benchmark', 'both']:
        print("\n🏁 Starting Benchmark Comparison...")
        print("-" * 40)
        
        benchmark = SchedulerBenchmark(base_dir=args.base_dir, dataset=args.dataset)
        
        # Define schedulers to benchmark
        benchmark_schedulers = args.benchmark_schedulers
        if benchmark_schedulers is None:
            # Default set of schedulers for benchmarking
            benchmark_schedulers = [
                'cosine_annealing', 'linear_decay', 'exponential_decay', 'step_decay',
                'polynomial_decay', 'sigmoid_decay', 'warmup_cosine', 'constant',
                'multi_cycle_cosine', 'cosine_with_restarts'
            ]
        
        try:
            benchmark_results = benchmark.run_scheduler_comparison(
                schedulers=benchmark_schedulers,
                epochs=args.epochs,
                trials=args.trials
            )
            results['benchmark'] = benchmark_results
            
            # Save benchmark results
            bench_output_file = output_dir / f"benchmark_{args.dataset}_{int(time.time())}.json"
            with open(bench_output_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            print(f"💾 Benchmark results saved to: {bench_output_file}")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            results['benchmark'] = {'error': str(e)}
    
    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 Analysis Complete!")
    print("=" * 60)
    print(f"⏱️ Total Time: {total_time:.2f}s")
    
    # Print optimization summary
    if 'optimization' in results and 'results' in results['optimization']:
        opt_results = results['optimization']['results']
        successful_opts = [r for r in opt_results if 'best_accuracy' in r]
        
        if successful_opts:
            best_opt = max(successful_opts, key=lambda x: x['best_accuracy'])
            print(f"\n🏆 Best Optimization Result:")
            print(f"   Method: {best_opt['method']}")
            print(f"   Accuracy: {best_opt['best_accuracy']:.4f}")
            print(f"   Best Parameters:")
            for param, value in best_opt['best_params'].items():
                if isinstance(value, float):
                    print(f"     {param}: {value:.3f}")
                else:
                    print(f"     {param}: {value}")
    
    # Print benchmark summary
    if 'benchmark' in results and 'results' in results['benchmark']:
        bench_results = results['benchmark']['results']
        sorted_bench = sorted(bench_results.items(), 
                            key=lambda x: x[1].get('mean_accuracy', 0), reverse=True)
        
        if sorted_bench:
            best_bench = sorted_bench[0]
            print(f"\n🥇 Best Benchmark Result:")
            print(f"   Scheduler: {best_bench[0]}")
            print(f"   Mean Accuracy: {best_bench[1]['mean_accuracy']:.4f} ± {best_bench[1]['std_accuracy']:.4f}")
            print(f"   Max Accuracy: {best_bench[1]['max_accuracy']:.4f}")
    
    # Save combined results
    if results:
        combined_output_file = output_dir / f"combined_analysis_{args.dataset}_{int(time.time())}.json"
        combined_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': args.dataset,
            'mode': args.mode,
            'total_time': total_time,
            'parameters': {
                'optimization_steps': args.steps if args.mode in ['optimize', 'both'] else None,
                'benchmark_trials': args.trials if args.mode in ['benchmark', 'both'] else None,
                'benchmark_epochs': args.epochs if args.mode in ['benchmark', 'both'] else None
            },
            'results': results
        }
        
        with open(combined_output_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        print(f"💾 Combined analysis saved to: {combined_output_file}")
    
    print("\n" + "=" * 60)
    print("✅ Temperature Scheduler Analysis Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
