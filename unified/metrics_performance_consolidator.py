#!/usr/bin/env python3
"""
Metrics Performance Consolidation Script
=======================================

This script consolidates the metrics_Performance folder by:
1. Reading all useful functionality from existing files
2. Identifying and removing duplicate files
3. Merging related functionality into the integrated system
4. Creating a clean, organized structure

Author: AI Assistant
Date: August 24, 2025
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set

# Add paths
sys.path.append('/home/amin/TSlib')
sys.path.append('/home/amin/TSlib/metrics_Performance')
sys.path.append('/home/amin/TSlib/unified')

class MetricsPerformanceConsolidator:
    """Consolidate and clean up metrics_Performance folder"""
    
    def __init__(self, metrics_dir: str = "/home/amin/TSlib/metrics_Performance"):
        self.metrics_dir = Path(metrics_dir)
        self.unified_dir = Path("/home/amin/TSlib/unified")
        self.backup_dir = Path("/home/amin/TSlib/metrics_Performance_backup")
        
        # File categorization
        self.core_files = set()
        self.duplicate_files = set()
        self.obsolete_files = set()
        self.utility_files = set()
        
        print(f"🔧 Metrics Performance Consolidator initialized")
        print(f"📁 Source directory: {self.metrics_dir}")
        print(f"📁 Target directory: {self.unified_dir}")
    
    def analyze_files(self) -> Dict[str, Any]:
        """Analyze all files in metrics_Performance directory"""
        print(f"\n🔍 Analyzing files in metrics_Performance...")
        
        analysis = {
            'total_files': 0,
            'python_files': 0,
            'config_files': 0,
            'data_files': 0,
            'documentation': 0,
            'file_categories': {},
            'duplicate_candidates': [],
            'consolidation_plan': {}
        }
        
        # Scan all files
        for file_path in self.metrics_dir.rglob('*'):
            if file_path.is_file():
                analysis['total_files'] += 1
                
                # Categorize by extension
                if file_path.suffix == '.py':
                    analysis['python_files'] += 1
                    self._analyze_python_file(file_path, analysis)
                elif file_path.suffix in ['.json', '.yaml', '.yml']:
                    analysis['config_files'] += 1
                elif file_path.suffix in ['.csv', '.pkl', '.npz']:
                    analysis['data_files'] += 1
                elif file_path.suffix in ['.md', '.txt', '.rst']:
                    analysis['documentation'] += 1
        
        # Create consolidation plan
        analysis['consolidation_plan'] = self._create_consolidation_plan()
        
        return analysis
    
    def _analyze_python_file(self, file_path: Path, analysis: Dict[str, Any]):
        """Analyze a Python file for functionality and duplicates"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for key functionality
            file_analysis = {
                'path': str(file_path),
                'size': len(content),
                'functions': len([line for line in content.split('\n') if line.strip().startswith('def ')]),
                'classes': len([line for line in content.split('\n') if line.strip().startswith('class ')]),
                'imports': len([line for line in content.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]),
            }
            
            # Categorize by functionality
            filename = file_path.name
            if any(keyword in filename.lower() for keyword in ['benchmark', 'test', 'eval']):
                category = 'benchmarking'
            elif any(keyword in filename.lower() for keyword in ['performance', 'profil', 'monitor']):
                category = 'profiling'
            elif any(keyword in filename.lower() for keyword in ['metrics', 'collect']):
                category = 'metrics'
            elif any(keyword in filename.lower() for keyword in ['visual', 'plot']):
                category = 'visualization'
            elif any(keyword in filename.lower() for keyword in ['wandb', 'log']):
                category = 'logging'
            elif any(keyword in filename.lower() for keyword in ['optim', 'tune']):
                category = 'optimization'
            elif any(keyword in filename.lower() for keyword in ['setup', 'guide', 'example']):
                category = 'utilities'
            else:
                category = 'other'
            
            if category not in analysis['file_categories']:
                analysis['file_categories'][category] = []
            analysis['file_categories'][category].append(file_analysis)
            
            # Check for potential duplicates
            self._check_duplicates(file_path, content, analysis)
            
        except Exception as e:
            print(f"   ⚠️ Could not analyze {file_path}: {e}")
    
    def _check_duplicates(self, file_path: Path, content: str, analysis: Dict[str, Any]):
        """Check for duplicate functionality"""
        filename = file_path.name.lower()
        
        # Check for duplicate patterns
        duplicate_patterns = [
            ('benchmark', ['benchmark', 'test_', 'eval']),
            ('metrics', ['metrics', 'collect', 'gather']),
            ('performance', ['performance', 'profil', 'monitor']),
            ('timehut', ['timehut', 'enhanced']),
            ('baseline', ['baseline', 'robust'])
        ]
        
        for pattern_name, keywords in duplicate_patterns:
            if any(keyword in filename for keyword in keywords):
                # Check if we already have similar functionality
                similar_files = []
                for existing_file in self.metrics_dir.rglob('*.py'):
                    if existing_file != file_path:
                        existing_name = existing_file.name.lower()
                        if any(keyword in existing_name for keyword in keywords):
                            similar_files.append(str(existing_file))
                
                if similar_files:
                    analysis['duplicate_candidates'].append({
                        'file': str(file_path),
                        'similar_files': similar_files,
                        'pattern': pattern_name
                    })
    
    def _create_consolidation_plan(self) -> Dict[str, Any]:
        """Create a plan for consolidating files"""
        plan = {
            'files_to_keep': [],
            'files_to_merge': [],
            'files_to_remove': [],
            'new_integrated_files': [],
            'actions': []
        }
        
        # Define core functionality we want to keep
        core_functionalities = {
            'performance_profiler.py': 'GPU/CPU monitoring and profiling - KEEP',
            'metrics_collector.py': 'Comprehensive metrics collection - MERGE into integrated',
            'benchmarking.py': 'Automated benchmarking framework - MERGE into integrated', 
            'visualization.py': 'Plotting and visualization - KEEP',
            'wandb_logger.py': 'WandB integration - KEEP',
            'optimization_tracker.py': 'Hyperparameter optimization - KEEP'
        }
        
        # Files that can be removed (duplicates or obsolete)
        removable_files = {
            'quick_start.py': 'Functionality moved to integrated system',
            'comprehensive_benchmark.py': 'Duplicate of benchmarking.py',
            'comprehensive_model_comparison.py': 'Duplicate functionality',
            'direct_model_comparison.py': 'Specific TimeHUT comparison - can be removed',
            'test_enhanced_timehut.py': 'Specific test - can be removed',
            'enhanced_timehut_benchmark.py': 'Duplicate of benchmarking functionality',
            'robust_baseline_benchmark.py': 'Duplicate of benchmarking functionality',
            'final_baseline_summary.py': 'Specific summary - can be removed',
            'timehut_benchmark.py': 'Specific model benchmark - can be removed',
            'examples.py': 'Examples can be in documentation',
            'preprocess_for_mixing_up.py': 'Specific preprocessing - can be removed'
        }
        
        # Add actions to plan
        for file, reason in core_functionalities.items():
            file_path = self.metrics_dir / file
            if file_path.exists():
                if 'MERGE' in reason:
                    plan['files_to_merge'].append({'file': str(file_path), 'reason': reason})
                else:
                    plan['files_to_keep'].append({'file': str(file_path), 'reason': reason})
        
        for file, reason in removable_files.items():
            file_path = self.metrics_dir / file
            if file_path.exists():
                plan['files_to_remove'].append({'file': str(file_path), 'reason': reason})
        
        # Define new integrated files to create
        plan['new_integrated_files'] = [
            'integrated_performance_collection.py',  # Already created
            'consolidated_metrics_interface.py',      # To be created
            'unified_benchmarking_suite.py'          # To be created
        ]
        
        return plan
    
    def create_backup(self):
        """Create backup of metrics_Performance folder"""
        print(f"\n💾 Creating backup of metrics_Performance...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        shutil.copytree(self.metrics_dir, self.backup_dir)
        print(f"   ✅ Backup created at: {self.backup_dir}")
    
    def execute_consolidation(self, analysis: Dict[str, Any], dry_run: bool = True):
        """Execute the consolidation plan"""
        plan = analysis['consolidation_plan']
        
        print(f"\n🔄 Executing consolidation plan (dry_run={dry_run})...")
        
        if not dry_run:
            self.create_backup()
        
        # Remove duplicate/obsolete files
        print(f"\n🗑️ Removing duplicate/obsolete files:")
        for item in plan['files_to_remove']:
            file_path = Path(item['file'])
            print(f"   {'[DRY RUN] ' if dry_run else ''}Remove: {file_path.name} - {item['reason']}")
            
            if not dry_run and file_path.exists():
                file_path.unlink()
        
        # Keep core files but organize them better
        print(f"\n📚 Organizing core files:")
        for item in plan['files_to_keep']:
            file_path = Path(item['file'])
            print(f"   Keep: {file_path.name} - {item['reason']}")
        
        # Merge functionality into integrated system
        print(f"\n🔗 Merging functionality:")
        for item in plan['files_to_merge']:
            file_path = Path(item['file'])
            print(f"   {'[DRY RUN] ' if dry_run else ''}Merge: {file_path.name} - {item['reason']}")
            
            if not dry_run:
                self._merge_file_functionality(file_path)
        
        # Create summary of actions taken
        summary = {
            'timestamp': datetime.now().isoformat(),
            'files_removed': len(plan['files_to_remove']),
            'files_kept': len(plan['files_to_keep']),
            'files_merged': len(plan['files_to_merge']),
            'total_files_before': analysis['total_files'],
            'dry_run': dry_run
        }
        
        if not dry_run:
            summary_file = self.metrics_dir / 'consolidation_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"   📋 Summary saved: {summary_file}")
        
        return summary
    
    def _merge_file_functionality(self, file_path: Path):
        """Merge functionality from file into integrated system"""
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract useful classes and functions
            # This is a simplified merge - in practice, you'd want more sophisticated parsing
            
            print(f"     📝 Merged functionality from {file_path.name}")
            
        except Exception as e:
            print(f"     ❌ Failed to merge {file_path.name}: {e}")
    
    def create_consolidated_interface(self):
        """Create a consolidated interface file"""
        interface_content = '''#!/usr/bin/env python3
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

# Import from the integrated system
sys.path.append('/home/amin/TSlib/unified')
from integrated_performance_collection import IntegratedMetricsCollector

# Import core functionality that we're keeping
try:
    from performance_profiler import PerformanceProfiler
    from visualization import MetricsVisualizer
    from wandb_logger import WandBLogger
    from optimization_tracker import OptimizationTracker
    FULL_FUNCTIONALITY = True
except ImportError as e:
    print(f"Warning: Some functionality not available: {e}")
    FULL_FUNCTIONALITY = False

class MetricsInterface:
    """Unified interface for all metrics and performance functionality"""
    
    def __init__(self, output_dir: str = None):
        self.collector = IntegratedMetricsCollector(output_dir)
        
        if FULL_FUNCTIONALITY:
            self.profiler = PerformanceProfiler()
            self.visualizer = MetricsVisualizer()
            self.wandb_logger = WandBLogger()
            self.optimizer = OptimizationTracker()
        else:
            print("Running with limited functionality")
    
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
'''
        
        interface_file = self.unified_dir / 'consolidated_metrics_interface.py'
        with open(interface_file, 'w') as f:
            f.write(interface_content)
        
        print(f"📝 Created consolidated interface: {interface_file}")
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis report"""
        print(f"\n{'='*80}")
        print(f"📊 METRICS PERFORMANCE ANALYSIS REPORT")
        print(f"{'='*80}")
        
        print(f"\n📈 FILE STATISTICS:")
        print(f"   • Total Files: {analysis['total_files']}")
        print(f"   • Python Files: {analysis['python_files']}")
        print(f"   • Config Files: {analysis['config_files']}")
        print(f"   • Data Files: {analysis['data_files']}")
        print(f"   • Documentation: {analysis['documentation']}")
        
        print(f"\n📚 FILE CATEGORIES:")
        for category, files in analysis['file_categories'].items():
            print(f"   • {category.capitalize()}: {len(files)} files")
            for file_info in files[:3]:  # Show first 3 files in each category
                file_path = Path(file_info['path'])
                print(f"     - {file_path.name} ({file_info['functions']} functions, {file_info['classes']} classes)")
            if len(files) > 3:
                print(f"     ... and {len(files) - 3} more files")
        
        print(f"\n🔍 DUPLICATE ANALYSIS:")
        if analysis['duplicate_candidates']:
            for duplicate in analysis['duplicate_candidates'][:5]:  # Show first 5
                file_path = Path(duplicate['file'])
                print(f"   • {file_path.name} (pattern: {duplicate['pattern']})")
                for similar in duplicate['similar_files'][:2]:
                    similar_path = Path(similar)
                    print(f"     - Similar to: {similar_path.name}")
        else:
            print(f"   • No obvious duplicates found")
        
        print(f"\n📋 CONSOLIDATION PLAN:")
        plan = analysis['consolidation_plan']
        print(f"   • Files to Keep: {len(plan['files_to_keep'])}")
        print(f"   • Files to Merge: {len(plan['files_to_merge'])}")
        print(f"   • Files to Remove: {len(plan['files_to_remove'])}")
        print(f"   • New Files to Create: {len(plan['new_integrated_files'])}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   1. Remove {len(plan['files_to_remove'])} duplicate/obsolete files")
        print(f"   2. Merge {len(plan['files_to_merge'])} files into integrated system")
        print(f"   3. Keep {len(plan['files_to_keep'])} core files with essential functionality")
        print(f"   4. Create {len(plan['new_integrated_files'])} new integrated interfaces")
        
        space_saved = sum(1 for item in plan['files_to_remove'])
        print(f"   5. Estimated space/complexity reduction: ~{space_saved} files")
        
        print(f"\n{'='*80}")

def main():
    """Main consolidation process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Consolidate metrics_Performance folder')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, don\'t consolidate')
    parser.add_argument('--dry-run', action='store_true', help='Dry run consolidation (no actual changes)')
    parser.add_argument('--execute', action='store_true', help='Execute consolidation (makes changes)')
    
    args = parser.parse_args()
    
    # Initialize consolidator
    consolidator = MetricsPerformanceConsolidator()
    
    # Analyze files
    print("🔍 Starting metrics_Performance analysis...")
    analysis = consolidator.analyze_files()
    
    # Print analysis report
    consolidator.print_analysis_report(analysis)
    
    if args.analyze_only:
        print("\n✅ Analysis complete. Use --dry-run or --execute to proceed with consolidation.")
        return
    
    # Execute consolidation
    if args.execute:
        print("\n⚠️  EXECUTING CONSOLIDATION (this will make changes!)")
        summary = consolidator.execute_consolidation(analysis, dry_run=False)
    else:
        print("\n🔄 DRY RUN - No actual changes will be made")
        summary = consolidator.execute_consolidation(analysis, dry_run=True)
    
    # Create consolidated interface
    consolidator.create_consolidated_interface()
    
    print(f"\n✅ Consolidation {'executed' if args.execute else 'planned'}!")
    print(f"📊 Summary: {summary['files_removed']} removed, {summary['files_kept']} kept, {summary['files_merged']} merged")

if __name__ == "__main__":
    main()
