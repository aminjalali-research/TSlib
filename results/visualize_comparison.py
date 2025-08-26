#!/usr/bin/env python3
"""
Quick Summary of TimeHUT vs Regular TS2Vec Comparison Results
"""

import json
from pathlib import Path

# Results data
results = {
    'TimeHUT Baseline': {'accuracy': 0.1333, 'auprc': 0.3233, 'time': 9.94, 'batch_size': 64},
    'TimeHUT AMC': {'accuracy': 0.0667, 'auprc': 0.2661, 'time': 11.93, 'batch_size': 64},
    'Regular TS2Vec': {'accuracy': 0.1333, 'auprc': 0.3100, 'time': 3.09, 'batch_size': 8}
}

def create_detailed_table():
    """Create a detailed comparison table"""
    print("\n" + "="*80)
    print("📋 DETAILED METRICS TABLE")
    print("="*80)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'AUPRC':<8} {'Time(s)':<8} {'Batch':<6} {'Efficiency':<12}")
    print("-" * 80)
    
    for model in results.keys():
        data = results[model]
        efficiency = data['accuracy'] / data['time']
        print(f"{model:<20} {data['accuracy']:<10.4f} {data['auprc']:<8.4f} {data['time']:<8.2f} {data['batch_size']:<6} {efficiency:<12.4f}")

def show_summary():
    """Show comparison summary"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    auprcs = [results[model]['auprc'] for model in models]
    times = [results[model]['time'] for model in models]
    
    print("="*60)
    print("📊 PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    # Find winners
    best_acc_models = [m for m in models if results[m]['accuracy'] == max(accuracies)]
    best_auprc_model = max(models, key=lambda m: results[m]['auprc'])
    fastest_model = min(models, key=lambda m: results[m]['time'])
    most_efficient_model = max(models, key=lambda m: results[m]['accuracy'] / results[m]['time'])
    
    print(f"🎯 Best Accuracy: {', '.join(best_acc_models)} ({max(accuracies):.4f})")
    print(f"📈 Best AUPRC: {best_auprc_model} ({results[best_auprc_model]['auprc']:.4f})")
    print(f"⚡ Fastest Training: {fastest_model} ({results[fastest_model]['time']:.2f}s)")
    print(f"🏆 Most Efficient: {most_efficient_model} ({results[most_efficient_model]['accuracy'] / results[most_efficient_model]['time']:.4f} acc/sec)")
    
    print(f"\n💡 Key Insights:")
    print(f"   • TimeHUT Baseline achieves best AUPRC (+4.3% over Regular TS2Vec)")
    print(f"   • Regular TS2Vec is 3.2x faster with equivalent accuracy")
    print(f"   • AMC optimization needs parameter tuning for this dataset")
    print(f"   • Batch size difference (64 vs 8) significantly impacts training time")

if __name__ == "__main__":
    # Show comparison summary
    show_summary()
    
    # Create detailed table
    create_detailed_table()
    
    print(f"\n📁 Full comparison report: /home/amin/TSlib/results/TimeHUT_vs_Regular_TS2Vec_Comparison_Report.md")
    print(f"📊 Detailed metrics JSON: /home/amin/TSlib/results/comprehensive_model_comparison_metrics.json")
