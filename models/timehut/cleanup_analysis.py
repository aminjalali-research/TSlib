#!/usr/bin/env python3
"""
TimeHUT Integration Cleanup Summary
==================================

This script documents the integration status and helps identify redundant files
that can be safely removed after successful integration of optimization functionality.

Integration Status:
✅ optimize_enhanced_schedulers.py → Integrated into temperature_schedulers.py (REMOVED)
✅ Enhanced parameter support → Added to train_unified_comprehensive.py
✅ Comprehensive optimization script → Created run_scheduler_optimization.py

File Status Analysis:
=====================

REMOVED (Already Done):
- optimize_enhanced_schedulers.py ✅ → Functionality moved to temperature_schedulers.py

POTENTIALLY REDUNDANT:
The following files may be redundant based on functionality overlap:

1. optimization/simple_optimization.py
   - Earlier optimization approach for temperature schedulers
   - Superseded by integrated optimization in temperature_schedulers.py

2. optimization/optimize_individual_method.py
   - Individual method optimization (likely superseded)
   - Check if functionality is covered by run_scheduler_optimization.py

3. optimization/timehut_optimization_suite.py
   - May overlap with run_scheduler_optimization.py functionality

4. comprehensive_analysis.py
   - Large analysis script (677 lines) - review to see if superseded
   - May still be needed for specific analysis tasks

5. baseline_comprehensive_benchmark.py
   - Baseline benchmarking (595 lines) - may be independent functionality
   - Keep unless specifically replaced

KEEP (Core/Independent Functionality):
- train_unified_comprehensive.py ✅ (Enhanced with new parameters)
- temperature_schedulers.py ✅ (Enhanced with optimization)
- run_scheduler_optimization.py ✅ (New comprehensive script)
- HUTGuide.md ✅ (Documentation)
- ts2vec.py (Core model)
- datautils.py (Core utilities)
- utils.py (Core utilities)

Action Plan:
============

Safe to Remove (if confirmed redundant):
1. Check optimization/*.py files for unique functionality
2. Archive or remove redundant optimization scripts
3. Keep any scripts with unique analysis capabilities
4. Maintain core training and utility scripts

Recommendations:
- Archive optimization/ directory contents to archive/ if redundant
- Keep run_scheduler_optimization.py as the primary optimization interface
- Maintain comprehensive_analysis.py if it has unique analysis features
- Keep baseline_comprehensive_benchmark.py unless specifically replaced
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict

def analyze_file_sizes():
    """Analyze file sizes to understand code complexity"""
    timehut_dir = Path("/home/amin/TSlib/models/timehut")
    
    print("📊 File Size Analysis")
    print("=" * 40)
    
    files_to_check = [
        "train_unified_comprehensive.py",
        "temperature_schedulers.py", 
        "run_scheduler_optimization.py",
        "comprehensive_analysis.py",
        "baseline_comprehensive_benchmark.py"
    ]
    
    for file_name in files_to_check:
        file_path = timehut_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            print(f"{file_name:<35} {size:>8} bytes, {lines:>4} lines")
        else:
            print(f"{file_name:<35} {'NOT FOUND':>15}")

def analyze_optimization_directory():
    """Analyze optimization directory contents"""
    opt_dir = Path("/home/amin/TSlib/models/timehut/optimization")
    
    print("\n📁 Optimization Directory Analysis")
    print("=" * 40)
    
    if not opt_dir.exists():
        print("Optimization directory not found")
        return
    
    py_files = list(opt_dir.glob("*.py"))
    for py_file in py_files:
        try:
            size = py_file.stat().st_size
            with open(py_file, 'r') as f:
                lines = len(f.readlines())
            print(f"{py_file.name:<35} {size:>8} bytes, {lines:>4} lines")
        except Exception as e:
            print(f"{py_file.name:<35} {'ERROR':>15}")

def suggest_cleanup_actions():
    """Suggest specific cleanup actions"""
    print("\n🧹 Suggested Cleanup Actions")
    print("=" * 40)
    
    actions = [
        {
            'action': 'KEEP',
            'files': [
                'train_unified_comprehensive.py',
                'temperature_schedulers.py',
                'run_scheduler_optimization.py',
                'HUTGuide.md',
                'ts2vec.py',
                'datautils.py',
                'utils.py'
            ],
            'reason': 'Core functionality with latest enhancements'
        },
        {
            'action': 'REVIEW',
            'files': [
                'comprehensive_analysis.py',
                'baseline_comprehensive_benchmark.py'
            ],
            'reason': 'Large scripts - check for unique functionality'
        },
        {
            'action': 'CONSIDER_ARCHIVING',
            'files': [
                'optimization/simple_optimization.py',
                'optimization/optimize_individual_method.py',
                'optimization/timehut_optimization_suite.py'
            ],
            'reason': 'Potentially superseded by integrated optimization'
        }
    ]
    
    for action_group in actions:
        print(f"\n{action_group['action']}:")
        print(f"  Reason: {action_group['reason']}")
        for file_name in action_group['files']:
            print(f"  - {file_name}")

def create_archive_directory():
    """Create archive directory for old scripts"""
    archive_dir = Path("/home/amin/TSlib/models/timehut/archive")
    archive_dir.mkdir(exist_ok=True)
    return archive_dir

if __name__ == "__main__":
    print("🔍 TimeHUT Integration Cleanup Analysis")
    print("=" * 60)
    
    analyze_file_sizes()
    analyze_optimization_directory()
    suggest_cleanup_actions()
    
    print(f"\n✅ Integration Status:")
    print("- optimize_enhanced_schedulers.py: REMOVED ✅")
    print("- temperature_schedulers.py: ENHANCED ✅") 
    print("- train_unified_comprehensive.py: ENHANCED ✅")
    print("- run_scheduler_optimization.py: CREATED ✅")
    
    print(f"\n📋 Next Steps:")
    print("1. Review suggested files for unique functionality")
    print("2. Archive redundant optimization scripts")
    print("3. Test integrated optimization system")
    print("4. Update documentation if needed")
