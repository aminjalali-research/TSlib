#!/usr/bin/env python3
"""
TSlib Unified Models Performance Analysis
========================================

This script uses the integrated metrics collector to analyze all models
in the unified folder and generate comprehensive performance reports.

Usage:
    python unified_models_performance_analysis.py --all-models
    python unified_models_performance_analysis.py --models TS2vec TimeHUT --datasets Chinatown
    python unified_models_performance_analysis.py --scheduler-comparison
    python unified_models_performance_analysis.py --production-assessment

Author: AI Assistant  
Date: August 2025
"""

import sys
import os
from pathlib import Path

# Add unified directory to path
unified_dir = Path("/home/amin/TSlib/unified")
sys.path.append(str(unified_dir))

from integrated_metrics_collector import IntegratedMetricsCollector

class UnifiedModelsAnalyzer:
    """Analyze all models in the unified TSlib system"""
    
    def __init__(self):
        self.collector = IntegratedMetricsCollector()
        
        # Define all available models from the unified system
        self.available_models = [
            'TS2vec', 'TimeHUT', 'SoftCLT', 'BIOT', 'VQ_MTM', 
            'DCRNN', 'Ti_MAE', 'SimMTM', 'TimesNet', 'TNC', 
            'CPC', 'CoST', 'MF-CLR', 'ConvTran', 'InceptionTime'
        ]
        
        # Define available datasets
        self.available_datasets = [
            'Chinatown', 'AtrialFibrillation', 'CricketX', 
            'EigenWorms', 'MotorImagery'
        ]
        
        # Fast models for quick testing
        self.fast_models = ['TS2vec', 'TimeHUT', 'MF-CLR', 'TNC']
        
        # Small datasets for quick testing  
        self.small_datasets = ['Chinatown', 'AtrialFibrillation']
    
    def run_comprehensive_analysis(self, models=None, datasets=None):
        """Run comprehensive performance analysis"""
        
        if models is None:
            models = self.fast_models  # Start with fast models
        if datasets is None:
            datasets = self.small_datasets  # Start with small datasets
        
        print("🚀 Starting comprehensive models analysis...")
        print(f"📊 Models: {models}")
        print(f"📁 Datasets: {datasets}")
        
        # Collect metrics
        results = self.collector.collect_comprehensive_metrics(models, datasets)
        
        # Print summary
        self.collector.print_summary()
        
        # Save results
        self.collector.save_results()
        
        return results
    
    def run_scheduler_comparison_study(self, models=None, datasets=None):
        """Run learning rate scheduler comparison study"""
        
        if models is None:
            models = ['TS2vec', 'TimeHUT']  # Fast models for scheduler testing
        if datasets is None:
            datasets = ['Chinatown']  # Small dataset for quick comparison
        
        print("🔄 Starting scheduler comparison study...")
        print(f"📊 Models: {models}")
        print(f"📁 Datasets: {datasets}")
        
        # Compare schedulers
        scheduler_results = self.collector.compare_schedulers_across_models(models, datasets)
        
        # Analyze results
        self._analyze_scheduler_results(scheduler_results)
        
        return scheduler_results
    
    def _analyze_scheduler_results(self, scheduler_results):
        """Analyze scheduler comparison results"""
        
        print("\n📊 SCHEDULER COMPARISON ANALYSIS")
        print("="*60)
        
        for key, results in scheduler_results.items():
            print(f"\n🎯 {key}")
            print("-" * 30)
            
            # Sort schedulers by accuracy
            sorted_schedulers = sorted(results.items(), 
                                     key=lambda x: x[1].accuracy, 
                                     reverse=True)
            
            for i, (scheduler, metrics) in enumerate(sorted_schedulers):
                status_icon = "✅" if metrics.status == "success" else "❌"
                print(f"   {i+1}. {scheduler}: {status_icon} "
                      f"Acc={metrics.accuracy:.3f}, Time={metrics.total_time_s:.1f}s")
            
            if sorted_schedulers:
                best_scheduler, best_metrics = sorted_schedulers[0]
                print(f"   🏆 Winner: {best_scheduler} ({best_metrics.accuracy:.3f})")
    
    def run_production_readiness_assessment(self, models=None, datasets=None):
        """Run production deployment readiness assessment"""
        
        if models is None:
            models = self.fast_models
        if datasets is None:
            datasets = self.small_datasets
        
        print("📋 Starting production readiness assessment...")
        
        # First collect comprehensive metrics
        results = self.collector.collect_comprehensive_metrics(models, datasets)
        
        # Analyze production readiness
        self._analyze_production_readiness()
        
        return results
    
    def _analyze_production_readiness(self):
        """Analyze production readiness from collected metrics"""
        
        print("\n📋 PRODUCTION READINESS ASSESSMENT")
        print("="*60)
        
        production_ready = []
        needs_optimization = []
        not_ready = []
        
        for metrics in self.collector.all_metrics:
            if metrics.status == "success" and 'production_readiness' in metrics.environment:
                assessment = metrics.environment['production_readiness']
                level = assessment['readiness_level']
                
                model_info = {
                    'model': metrics.model_name,
                    'dataset': metrics.dataset,
                    'accuracy': metrics.accuracy,
                    'time': metrics.total_time_s,
                    'score': assessment['overall_score']
                }
                
                if level == 'Production Ready':
                    production_ready.append(model_info)
                elif 'Optimization' in level:
                    needs_optimization.append(model_info)
                else:
                    not_ready.append(model_info)
        
        # Report results
        if production_ready:
            print(f"\n✅ PRODUCTION READY ({len(production_ready)} models):")
            for info in sorted(production_ready, key=lambda x: x['score'], reverse=True):
                print(f"   • {info['model']} on {info['dataset']}: "
                      f"Score={info['score']:.2f}, Acc={info['accuracy']:.3f}")
        
        if needs_optimization:
            print(f"\n⚠️ NEEDS OPTIMIZATION ({len(needs_optimization)} models):")
            for info in sorted(needs_optimization, key=lambda x: x['score'], reverse=True):
                print(f"   • {info['model']} on {info['dataset']}: "
                      f"Score={info['score']:.2f}, Acc={info['accuracy']:.3f}")
        
        if not_ready:
            print(f"\n❌ NOT PRODUCTION READY ({len(not_ready)} models):")
            for info in sorted(not_ready, key=lambda x: x['score'], reverse=True):
                print(f"   • {info['model']} on {info['dataset']}: "
                      f"Score={info['score']:.2f}, Acc={info['accuracy']:.3f}")
    
    def run_model_efficiency_analysis(self, models=None, datasets=None):
        """Analyze model efficiency (accuracy vs time vs memory)"""
        
        if models is None:
            models = self.fast_models
        if datasets is None:
            datasets = self.small_datasets
        
        print("⚡ Starting model efficiency analysis...")
        
        # Collect metrics
        results = self.collector.collect_comprehensive_metrics(models, datasets)
        
        # Analyze efficiency
        self._analyze_model_efficiency()
        
        return results
    
    def _analyze_model_efficiency(self):
        """Analyze model efficiency metrics"""
        
        print("\n⚡ MODEL EFFICIENCY ANALYSIS")
        print("="*60)
        
        successful_metrics = [m for m in self.collector.all_metrics if m.status == "success"]
        
        if not successful_metrics:
            print("No successful runs to analyze")
            return
        
        print(f"\n📊 EFFICIENCY RANKINGS:")
        print("Model".ljust(12) + "Dataset".ljust(15) + "Acc".ljust(8) + "Time(s)".ljust(10) + "Mem(GB)".ljust(10) + "Efficiency")
        print("-" * 70)
        
        # Sort by a composite efficiency score (accuracy / time)
        efficiency_scores = []
        for m in successful_metrics:
            if m.total_time_s > 0:
                efficiency = m.accuracy / m.total_time_s  # accuracy per second
                efficiency_scores.append((m, efficiency))
        
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (m, eff_score) in enumerate(efficiency_scores[:10]):
            print(f"{m.model_name:<12}{m.dataset:<15}{m.accuracy:<8.3f}"
                  f"{m.total_time_s:<10.1f}{m.peak_gpu_memory_gb:<10.2f}{eff_score:.4f}")
        
        # Best performers in each category
        if successful_metrics:
            best_accuracy = max(successful_metrics, key=lambda m: m.accuracy)
            fastest = min(successful_metrics, key=lambda m: m.total_time_s)
            most_memory_efficient = min([m for m in successful_metrics if m.peak_gpu_memory_gb > 0], 
                                      key=lambda m: m.peak_gpu_memory_gb, default=successful_metrics[0])
            
            print(f"\n🏆 CATEGORY WINNERS:")
            print(f"   • Best Accuracy: {best_accuracy.model_name} ({best_accuracy.accuracy:.3f})")
            print(f"   • Fastest: {fastest.model_name} ({fastest.total_time_s:.1f}s)")
            print(f"   • Most Memory Efficient: {most_memory_efficient.model_name} "
                  f"({most_memory_efficient.peak_gpu_memory_gb:.2f}GB)")
    
    def run_quick_test(self):
        """Run quick test with subset of models and datasets"""
        
        print("🔍 Running quick test with fastest models...")
        
        # Use fastest models and smallest datasets
        quick_models = ['TS2vec', 'MF-CLR']
        quick_datasets = ['Chinatown']
        
        results = self.run_comprehensive_analysis(quick_models, quick_datasets)
        
        print("✅ Quick test completed!")
        return results
    
    def run_full_analysis(self):
        """Run complete analysis with all models and datasets"""
        
        print("🚀 Running full analysis with all available models...")
        print("⚠️ This may take several hours to complete!")
        
        # Use all available models and datasets
        results = self.run_comprehensive_analysis(
            self.available_models, 
            self.available_datasets
        )
        
        # Also run scheduler comparison on subset
        scheduler_results = self.run_scheduler_comparison_study()
        
        print("✅ Full analysis completed!")
        return results, scheduler_results

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TSlib Unified Models Performance Analysis'
    )
    
    # Analysis modes
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with subset of models')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Run complete analysis (may take hours)')
    parser.add_argument('--scheduler-comparison', action='store_true',
                       help='Run scheduler comparison study')
    parser.add_argument('--production-assessment', action='store_true',
                       help='Run production readiness assessment')
    parser.add_argument('--efficiency-analysis', action='store_true',
                       help='Run model efficiency analysis')
    
    # Model and dataset selection
    parser.add_argument('--models', nargs='+',
                       help='Specific models to analyze')
    parser.add_argument('--datasets', nargs='+', 
                       help='Specific datasets to test')
    parser.add_argument('--all-models', action='store_true',
                       help='Use all available models')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = UnifiedModelsAnalyzer()
    
    # Determine models and datasets
    models = None
    datasets = None
    
    if args.models:
        models = args.models
    elif args.all_models:
        models = analyzer.available_models
    
    if args.datasets:
        datasets = args.datasets
    
    # Run requested analyses
    if args.quick_test:
        analyzer.run_quick_test()
    
    elif args.full_analysis:
        analyzer.run_full_analysis()
    
    elif args.scheduler_comparison:
        analyzer.run_scheduler_comparison_study(models, datasets)
    
    elif args.production_assessment:
        analyzer.run_production_readiness_assessment(models, datasets)
    
    elif args.efficiency_analysis:
        analyzer.run_model_efficiency_analysis(models, datasets)
    
    else:
        # Default: run comprehensive analysis
        analyzer.run_comprehensive_analysis(models, datasets)
    
    print("\n🎉 Analysis complete! Check the results directory for detailed reports.")

if __name__ == "__main__":
    main()
