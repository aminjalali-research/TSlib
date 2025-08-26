#!/usr/bin/env python3
"""
Quick test of enhanced comprehensive benchmark before full run
"""

import os
import sys
sys.path.append('/home/amin/TSlib/models/timehut')

def test_single_model():
    """Test single model before full run"""
    
    from enhanced_baseline_comprehensive import EnhancedBaselineComprehensiveBenchmark
    
    print("🧪 Testing Enhanced Comprehensive Benchmark - Single Model")
    print("="*60)
    
    # Initialize runner
    runner = EnhancedBaselineComprehensiveBenchmark()
    
    # Override for quick test - only TS2vec with reduced epochs
    runner.config['epochs'] = 10  # Quick test
    runner.config['timeout'] = 600  # 10 minutes
    
    # Test only TS2vec
    test_models = {'TS2vec': runner.baseline_models['TS2vec']}
    runner.baseline_models = test_models
    
    try:
        # Run single model test
        results = runner.run_all_enhanced_benchmarks()
        
        # Print results
        runner.print_enhanced_performance_table()
        
        # Save results
        runner.save_enhanced_results()
        
        print("\n✅ Single model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Single model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_model()
    if success:
        print("\n✅ Ready to run full enhanced comprehensive benchmark!")
        print("Run: python enhanced_baseline_comprehensive.py")
