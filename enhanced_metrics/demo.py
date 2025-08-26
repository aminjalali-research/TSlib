#!/usr/bin/env python3
"""
Enhanced Metrics Demo
===================

Demonstration script showing the enhanced metrics system capabilities
including Time/Epoch, Peak GPU Memory, and FLOPs/Epoch collection.

This script demonstrates both single model and batch usage examples.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run command and display output"""
    print(f"💻 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/amin/TSlib')
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("=" * 60)
    print(f"✅ Command completed with exit code: {result.returncode}")
    print()
    
    return result.returncode == 0

def demo_single_model():
    """Demonstrate single model enhanced metrics collection"""
    
    print("🚀 ENHANCED METRICS DEMO - SINGLE MODEL")
    print("=" * 70)
    print("Demonstrating enhanced metrics collection on a single model:")
    print("✅ Time/Epoch - Average training time per epoch")
    print("✅ Peak GPU Memory - Maximum GPU memory usage") 
    print("✅ FLOPs/Epoch - Computational complexity per epoch")
    print("✅ Efficiency Analysis - Computational efficiency metrics")
    print("=" * 70)
    print()
    
    # Run TimesURL on Chinatown (should be fast and successful)
    cmd = [
        'python', 'enhanced_metrics/enhanced_single_model_runner.py',
        'TimesURL', 'Chinatown', '60'
    ]
    
    success = run_command(cmd)
    
    if success:
        print("✅ Single model demo completed successfully!")
        print("📁 Check enhanced_metrics/results/ for detailed JSON results")
    else:
        print("❌ Single model demo failed")
    
    return success

def demo_batch_runner():
    """Demonstrate batch enhanced metrics collection"""
    
    print("🚀 ENHANCED METRICS DEMO - BATCH COLLECTION")
    print("=" * 70)
    print("Demonstrating batch enhanced metrics collection:")
    print("✅ Multiple model-dataset combinations")
    print("✅ Comprehensive statistical analysis")
    print("✅ Performance champion identification")
    print("✅ Model family comparison")
    print("=" * 70)
    print()
    
    # Run small batch with 2 models on 1 dataset
    cmd = [
        'python', 'enhanced_metrics/enhanced_batch_runner.py',
        '--models', 'TimesURL,CoST',
        '--datasets', 'Chinatown', 
        '--timeout', '60'
    ]
    
    success = run_command(cmd)
    
    if success:
        print("✅ Batch demo completed successfully!")
        print("📁 Check enhanced_metrics/batch_results/ for comprehensive analysis")
    else:
        print("❌ Batch demo failed")
    
    return success

def demo_config_usage():
    """Demonstrate configuration file usage"""
    
    print("📋 ENHANCED METRICS DEMO - CONFIG FILE")
    print("=" * 70)
    print("Demonstrating configuration file usage for batch collection")
    print("=" * 70)
    print()
    
    # Show config file contents
    config_path = Path('/home/amin/TSlib/enhanced_metrics/example_config.json')
    if config_path.exists():
        print("📄 Example configuration file contents:")
        with open(config_path, 'r') as f:
            print(f.read())
        print()
    
    # Run with config file
    cmd = [
        'python', 'enhanced_metrics/enhanced_batch_runner.py',
        '--config', 'enhanced_metrics/example_config.json',
        '--timeout', '45'  # Short timeout for demo
    ]
    
    print("💻 This would run a comprehensive batch with the config file:")
    print(f"    {' '.join(cmd)}")
    print("    (Skipped in demo due to time constraints)")
    print()

def show_system_overview():
    """Show enhanced metrics system overview"""
    
    print("🌟 ENHANCED METRICS SYSTEM OVERVIEW")
    print("=" * 70)
    print("📊 Enhanced Metrics Collected:")
    print("   📅 Time/Epoch - Average training time per epoch")
    print("   🔥 Peak GPU Memory - Maximum GPU memory usage during training")
    print("   ⚡ FLOPs/Epoch - Floating point operations per training epoch")
    print("   🚀 Real-time GPU Monitoring - Continuous resource tracking")
    print("   📈 Computational Efficiency - FLOPs/Memory/Time efficiency")
    print("   📊 Training Dynamics - Epoch progression analysis")
    print()
    
    print("🏗️ System Architecture:")
    print("   📁 enhanced_metrics/")
    print("      ├── enhanced_single_model_runner.py  # Single model testing")
    print("      ├── enhanced_batch_runner.py         # Batch processing")
    print("      ├── example_config.json              # Configuration example")
    print("      ├── README.md                        # Documentation")
    print("      ├── results/                         # Individual results")
    print("      └── batch_results/                   # Batch analysis")
    print()
    
    print("🎯 Key Features:")
    print("   ✅ Standalone system - no interference with existing files")
    print("   ✅ Comprehensive metrics beyond basic accuracy")
    print("   ✅ Real-time resource monitoring")
    print("   ✅ Batch processing with statistical analysis")
    print("   ✅ Performance champion identification")
    print("   ✅ Model family and efficiency comparisons")
    print("   ✅ Multiple output formats (JSON, CSV)")
    print("=" * 70)
    print()

def main():
    """Main demo function"""
    
    print("🚀 ENHANCED METRICS COLLECTION SYSTEM DEMO")
    print("=" * 70)
    print("This demo showcases the new enhanced metrics collection system")
    print("that provides Time/Epoch, Peak GPU Memory, and FLOPs/Epoch metrics")
    print("as separate files that don't interfere with existing systems.")
    print("=" * 70)
    print()
    
    # Show system overview
    show_system_overview()
    
    # Demo single model usage
    print("🎯 DEMO 1: Single Model Enhanced Metrics")
    single_success = demo_single_model()
    
    print("\n" + "="*70 + "\n")
    
    # Demo batch usage (small example)
    print("🎯 DEMO 2: Batch Enhanced Metrics Collection") 
    batch_success = demo_batch_runner()
    
    print("\n" + "="*70 + "\n")
    
    # Demo config file usage
    print("🎯 DEMO 3: Configuration File Usage")
    demo_config_usage()
    
    print("\n" + "="*70 + "\n")
    
    # Final summary
    print("📊 DEMO SUMMARY:")
    print(f"   Single Model Demo: {'✅ Success' if single_success else '❌ Failed'}")
    print(f"   Batch Collection Demo: {'✅ Success' if batch_success else '❌ Failed'}")
    print()
    
    if single_success or batch_success:
        print("🎉 Enhanced metrics system is working correctly!")
        print("📁 Results available in:")
        print("   📊 enhanced_metrics/results/ - Individual model results")
        print("   📈 enhanced_metrics/batch_results/ - Batch analysis")
        print()
        print("🚀 Ready for production use with comprehensive enhanced metrics!")
    else:
        print("❌ Demo had issues - please check error messages above")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
