#!/usr/bin/env python3
"""
Quick Test - Baseline Models Comprehensive Benchmarking
======================================================

Quick test version with reduced epochs to validate the approach before running full 200-epoch benchmark.
Uses same format and metrics as comprehensive_benchmark.py
"""

import os
import sys
sys.path.append('/home/amin/TSlib/models/timehut')

def quick_test_baseline_comprehensive():
    """Run a quick test with reduced epochs"""
    
    from baseline_comprehensive_benchmark import BaselineComprehensiveBenchmark
    
    print("🧪 Quick Test - Baseline Models Comprehensive Benchmark")
    print("Configuration: Batch Size=8, Epochs=5 (quick test)")
    print("="*70)
    
    # Initialize runner
    runner = BaselineComprehensiveBenchmark()
    
    # Override configuration for quick test
    runner.config['epochs'] = 5
    runner.config['timeout'] = 600  # 10 minutes timeout
    
    # Test with only TS2vec first (known working model)
    test_models = {'TS2vec': runner.baseline_models['TS2vec']}
    runner.baseline_models = test_models
    
    # Run quick test
    try:
        print("🔧 Running quick test with TS2vec only...")
        results = runner.run_all_benchmarks()
        
        # Print results
        runner.print_performance_table()
        
        # Save results
        runner.save_results()
        
        print("\n✅ Quick test completed successfully!")
        print("✅ Ready to run full 200-epoch comprehensive benchmark")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_comprehensive_benchmark():
    """Run the full comprehensive benchmark with all models and 200 epochs"""
    
    from baseline_comprehensive_benchmark import BaselineComprehensiveBenchmark
    
    print("🚀 Full Comprehensive Baseline Models Benchmark")
    print("Configuration: Batch Size=8, Epochs=200")
    print("="*80)
    
    # Initialize runner
    runner = BaselineComprehensiveBenchmark()
    
    # Run comprehensive benchmark
    try:
        results = runner.run_all_benchmarks()
        
        # Print comprehensive performance table
        runner.print_performance_table()
        
        # Save comprehensive results
        runner.save_results()
        
        print("\n🎉 Full comprehensive baseline benchmarking completed!")
        return results
        
    except Exception as e:
        print(f"❌ Full benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Models Comprehensive Benchmark")
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Run quick test (5 epochs) or full benchmark (200 epochs)')
    parser.add_argument('--model', type=str, help='Run specific model only')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        print("🧪 Running quick test...")
        success = quick_test_baseline_comprehensive()
        if success:
            print("\n📝 To run full benchmark:")
            print("   python baseline_comprehensive_benchmark.py --mode full")
    else:
        print("🚀 Running full comprehensive benchmark...")
        results = run_full_comprehensive_benchmark()

if __name__ == "__main__":
    main()
