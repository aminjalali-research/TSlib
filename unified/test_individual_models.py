#!/usr/bin/env python3
"""
Individual Model Tester - Debug and fix model execution issues
Following TaskGuide.md requirements for proper conda environment usage
"""

import subprocess
import json
import time
import os
import sys
from pathlib import Path

# Add TSlib to path for imports
sys.path.append('/home/amin/TSlib')
sys.path.append('/home/amin/TSlib/unified')

from master_benchmark_pipeline import MasterBenchmarkPipeline

def test_single_model_with_debug(model_name, dataset, environment):
    """Test a single model with full debug output"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {model_name} on {dataset}")
    print(f"🌍 Environment: {environment}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"{'='*80}")
    
    # Check if we're in the correct directory
    if not os.path.exists('/home/amin/TSlib'):
        print("❌ ERROR: TSlib directory not found!")
        return False
        
    # Change to TSlib directory
    original_dir = os.getcwd()
    os.chdir('/home/amin/TSlib')
    print(f"✅ Changed to: {os.getcwd()}")
    
    try:
        # Create benchmarker instance
        benchmarker = MasterBenchmarkPipeline()
        
        # Test the model
        result = benchmarker.run_single_model_benchmark(
            model_name=model_name,
            dataset=dataset
        )
        
        print(f"\n📊 RESULT: {model_name} on {dataset}")
        if result:
            print("✅ SUCCESS!")
            print(f"📈 Accuracy: {result.accuracy:.2f}%")
            print(f"📊 F1 Score: {result.f1_score:.3f}")
            success = True
        else:
            print("❌ FAILED!")
            success = False
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)

def main():
    """Test individual models following TaskGuide.md priorities"""
    
    # Test plan: Start with known working models, then debug failing ones
    test_cases = [
        # ✅ Known working models (based on previous results)
        ("TS2vec", "Chinatown", "tslib"),
        ("TimeHUT", "Chinatown", "tslib"),
        ("TNC", "Chinatown", "mfclr"),
        
        # 🔧 Debug these models that should work but are failing
        ("SoftCLT", "Chinatown", "tslib"),  # Symlink issue fixed
        ("BIOT", "AtrialFibrillation", "vq_mtm"),  # VQ-MTM on UEA (should work)
        ("VQ_MTM", "AtrialFibrillation", "vq_mtm"),  # Core VQ-MTM model
        
        # 🎯 High-priority failures to debug
        ("ConvTran", "Chinatown", "mfclr"),  # MF-CLR algorithm issue
        ("TFC", "Chinatown", "mfclr"),  # Parameter mismatch
        ("TimesNet", "AtrialFibrillation", "vq_mtm"),  # Architecture compatibility
    ]
    
    print("🚀 Starting Individual Model Testing")
    print(f"📍 Base directory: /home/amin/TSlib")
    
    results = {}
    
    for i, (model, dataset, env) in enumerate(test_cases, 1):
        print(f"\n{'🔄'*20} TEST {i}/{len(test_cases)} {'🔄'*20}")
        
        success = test_single_model_with_debug(model, dataset, env)
        results[f"{model}_{dataset}"] = success
        
        # Brief pause between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'🎯'*30} SUMMARY {'🎯'*30}")
    successes = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"✅ Successful: {successes}/{total}")
    print(f"❌ Failed: {total-successes}/{total}")
    
    print("\n📊 Detailed Results:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    # Save results
    results_file = "/home/amin/TSlib/results/individual_model_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total,
            'successful': successes,
            'failed': total - successes,
            'success_rate': f"{successes/total*100:.1f}%",
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    if successes < total:
        print(f"\n🔧 Next steps: Analyze failures and fix according to TaskGuide.md")
        return False
    else:
        print(f"\n🎉 All tests passed! System is working correctly.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
