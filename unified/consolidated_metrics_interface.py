#!/usr/bin/env python3
"""
Consolidated Metrics Performance Interface
=========================================

This module provides a unified interface to all metrics and performance
functionality that was previously scattered across multiple files.

Usage:
    from consolidated_metrics_interface import MetricsInterface
    
    # Initialize interface
    interface = MetricsInterface()
    
    # Run comprehensive analysis
    results = interface.run_comprehensive_analysis(models=['TS2vec', 'TimeHUT'])
    
    # Generate reports
    interface.generate_report(results)

Author: Consolidated from metrics_Performance folder
Date: August 24, 2025
"""

import sys
from pathlib import Path
from datetime import datetime

# Import from the integrated system
sys.path.append('/home/amin/TSlib/unified')
from integrated_performance_collection import IntegratedMetricsCollector

# Import core functionality that we're keeping
try:
    # Optional advanced functionality - graceful fallback if not available
    # sys.path.append('/home/amin/TSlib/metrics_Performance')
    # from performance_profiler import PerformanceProfiler
    # from visualization import MetricsVisualizer
    # from wandb_logger import WandBLogger
    # from optimization_tracker import OptimizationTracker
    FULL_FUNCTIONALITY = False  # Disabled - use enhanced_metrics instead
except ImportError as e:
    print(f"Warning: Some functionality not available: {e}")
    FULL_FUNCTIONALITY = False

class MetricsInterface:
    """Unified interface for all metrics and performance functionality"""
    
    def __init__(self, output_dir: str = None):
        self.collector = IntegratedMetricsCollector(output_dir)
        
        if FULL_FUNCTIONALITY:
            # Advanced functionality disabled - use enhanced_metrics for comprehensive analysis
            print("Advanced metrics functionality available")
        else:
            print("Using basic functionality - for comprehensive metrics use enhanced_metrics/")
    
    def run_comprehensive_analysis(self, models=None, datasets=None, 
                                 include_visualization=True, use_wandb=False):
        """Run comprehensive analysis with all available tools"""
        print("🚀 Running comprehensive metrics analysis...")
        
        # Collect comprehensive metrics
        results = self.collector.collect_comprehensive_metrics(
            models=models, datasets=datasets
        )
        
        # Generate visualizations if available
        if FULL_FUNCTIONALITY and include_visualization:
            print("📊 Generating visualizations...")
            # self.visualizer.create_comprehensive_plots(results)
        
        # Log to WandB if requested
        if FULL_FUNCTIONALITY and use_wandb:
            print("📡 Logging to WandB...")
            # self.wandb_logger.log_experiment_results(results)
        
        return results
    
    def run_quick_benchmark(self, models=None):
        """Quick benchmark equivalent to old quick_start.py"""
        print("⚡ Running quick benchmark...")
        
        return self.collector.collect_comprehensive_metrics(
            models=models, 
            datasets=['Chinatown', 'AtrialFibrillation'],
            include_schedulers=False,
            include_production_assessment=False
        )
    
    def generate_report(self, results, format='markdown'):
        """Generate comprehensive report"""
        self.collector._generate_comprehensive_report(
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )

# Convenience functions for backward compatibility
def quick_benchmark_all_models(**kwargs):
    """Backward compatible quick benchmark function"""
    interface = MetricsInterface()
    return interface.run_quick_benchmark()

def comprehensive_model_comparison(**kwargs):
    """Backward compatible comprehensive comparison"""
    interface = MetricsInterface()
    return interface.run_comprehensive_analysis()

# Main execution
if __name__ == "__main__":
    interface = MetricsInterface()
    results = interface.run_comprehensive_analysis()
    interface.generate_report(results)
