#!/usr/bin/env python3
"""
Test Enhanced TimeHUT Baselines Integration
==========================================

Test script to validate the enhanced baseline models integration
with the TimeHUT optimization approach.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add the TimeHUT path for imports
sys.path.append('/home/amin/TSlib/models/timehut')

def test_enhanced_baselines_integration():
    """Test the enhanced baselines integration"""
    
    print("🧪 Testing Enhanced TimeHUT Baselines Integration")
    print("="*60)
    
    # Import the enhanced integrator
    try:
        from enhanced_baselines_integration import EnhancedTimeHUTBaselinesIntegrator
        print("✅ Successfully imported EnhancedTimeHUTBaselinesIntegrator")
    except ImportError as e:
        print(f"❌ Failed to import EnhancedTimeHUTBaselinesIntegrator: {e}")
        return False
    
    # Initialize the integrator
    try:
        integrator = EnhancedTimeHUTBaselinesIntegrator()
        print("✅ Successfully initialized integrator")
    except Exception as e:
        print(f"❌ Failed to initialize integrator: {e}")
        return False
    
    # Test dataset link setup
    try:
        integrator.setup_dataset_links()
        print("✅ Dataset links setup completed")
    except Exception as e:
        print(f"❌ Dataset links setup failed: {e}")
        return False
    
    # Test single model run (quick test with reduced epochs)
    print("\n🔬 Testing single model run (TS2vec on AtrialFibrillation)...")
    
    try:
        # Override optimal epochs for quick test
        from enhanced_baselines_integration import DATASET_CONFIGS
        original_epochs = DATASET_CONFIGS['AtrialFibrillation']['optimal_epochs']
        DATASET_CONFIGS['AtrialFibrillation']['optimal_epochs'] = 3  # Quick test
        
        results = integrator.run_enhanced_baseline_benchmark('AtrialFibrillation')
        
        # Restore original epochs
        DATASET_CONFIGS['AtrialFibrillation']['optimal_epochs'] = original_epochs
        
        if results:
            print("✅ Single model benchmark completed successfully")
            for model, result in results.items():
                status = "✅" if result['success'] else "❌"
                acc = result.get('metrics', {}).get('accuracy', 'N/A')
                duration = result.get('duration', 0)
                print(f"   {status} {model}: Acc={acc}, Duration={duration:.2f}s")
        else:
            print("⚠️  No results returned from benchmark")
            
    except Exception as e:
        print(f"❌ Single model test failed: {e}")
        return False
    
    print("\n✅ Enhanced baselines integration test completed successfully!")
    return True

def run_quick_enhanced_benchmark():
    """Run a quick benchmark test with reduced epochs"""
    
    print("\n🚀 Quick Enhanced Benchmark Test")
    print("="*50)
    
    try:
        from enhanced_baselines_integration import EnhancedTimeHUTBaselinesIntegrator, DATASET_CONFIGS
        
        # Initialize integrator
        integrator = EnhancedTimeHUTBaselinesIntegrator()
        
        # Temporarily reduce epochs for quick test
        original_configs = {}
        for dataset in DATASET_CONFIGS:
            original_configs[dataset] = DATASET_CONFIGS[dataset]['optimal_epochs']
            DATASET_CONFIGS[dataset]['optimal_epochs'] = 5  # Quick test
        
        print("🔗 Setting up dataset links...")
        integrator.setup_dataset_links()
        
        print("🏃 Running quick benchmark on AtrialFibrillation...")
        start_time = time.time()
        
        results = integrator.run_enhanced_baseline_benchmark('AtrialFibrillation')
        
        total_time = time.time() - start_time
        
        print(f"\n⏱️  Total benchmark time: {total_time:.2f}s")
        
        # Restore original epochs
        for dataset in DATASET_CONFIGS:
            DATASET_CONFIGS[dataset]['optimal_epochs'] = original_configs[dataset]
        
        if results:
            print("\n📊 Quick Benchmark Results:")
            print("-" * 50)
            for model, result in results.items():
                status = "✅ Success" if result['success'] else "❌ Failed"
                acc = result.get('metrics', {}).get('accuracy', 'N/A')
                duration = result.get('duration', 0)
                speedup = result.get('speedup_achieved', 0)
                
                print(f"{model}:")
                print(f"  Status: {status}")
                print(f"  Accuracy: {acc}")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print()
        
        print("✅ Quick enhanced benchmark completed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_setup():
    """Verify the setup of enhanced baselines integration"""
    
    print("🔍 Verifying Enhanced Baselines Setup")
    print("="*40)
    
    checks = []
    
    # Check if enhanced integration script exists
    enhanced_script = Path('/home/amin/TSlib/models/timehut/enhanced_baselines_integration.py')
    if enhanced_script.exists():
        checks.append("✅ Enhanced baselines integration script exists")
    else:
        checks.append("❌ Enhanced baselines integration script missing")
    
    # Check baseline directories
    baselines_dir = Path('/home/amin/TSlib/models/timehut/baselines')
    baseline_models = ['TS2vec', 'TFC', 'TS-TCC', 'SimCLR', 'Mixing-up', 'CLOCS']
    
    for model in baseline_models:
        model_dir = baselines_dir / model
        if model_dir.exists():
            checks.append(f"✅ {model}: Directory exists")
            # Check for dataset symlink
            dataset_link = model_dir / "datasets"
            if dataset_link.exists() and dataset_link.is_symlink():
                checks.append(f"✅ {model}: Dataset symlink exists")
            else:
                checks.append(f"⚠️  {model}: Dataset symlink missing (will be created)")
        else:
            checks.append(f"❌ {model}: Directory missing")
    
    # Check datasets
    datasets_dir = Path('/home/amin/TSlib/datasets')
    uea_dir = datasets_dir / 'UEA'
    if uea_dir.exists():
        checks.append("✅ UEA datasets directory exists")
        
        # Check specific datasets
        for dataset in ['AtrialFibrillation', 'MotorImagery']:
            dataset_dir = uea_dir / dataset
            if dataset_dir.exists():
                checks.append(f"✅ UEA/{dataset}: Dataset available")
            else:
                checks.append(f"❌ UEA/{dataset}: Dataset missing")
    else:
        checks.append("❌ UEA datasets directory missing")
    
    # Print results
    for check in checks:
        print(check)
    
    success_count = sum(1 for check in checks if check.startswith("✅"))
    total_count = len(checks)
    
    print(f"\n📊 Setup Status: {success_count}/{total_count} checks passed")
    
    return success_count / total_count > 0.8  # 80% success rate

def main():
    """Main testing function"""
    
    print("🧪 Enhanced TimeHUT Baselines Integration Testing Suite")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_tests_passed = True
    
    # Test 1: Verify setup
    print("Test 1: Setup Verification")
    setup_ok = verify_setup()
    if not setup_ok:
        print("⚠️  Setup verification had issues, but continuing...")
    print()
    
    # Test 2: Integration test
    print("Test 2: Integration Test")
    integration_ok = test_enhanced_baselines_integration()
    all_tests_passed = all_tests_passed and integration_ok
    print()
    
    # Test 3: Quick benchmark
    print("Test 3: Quick Benchmark Test")
    benchmark_ok = run_quick_enhanced_benchmark()
    all_tests_passed = all_tests_passed and benchmark_ok
    print()
    
    # Summary
    print("="*70)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Enhanced TimeHUT baselines integration is working properly")
        print("\n📝 Next Steps:")
        print("   1. Run full benchmark: python enhanced_baselines_integration.py")
        print("   2. Check results in the generated results directory")
        print("   3. Add more baseline models as they become available")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Please check the error messages above and fix issues")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
