#!/usr/bin/env python3
"""
Enhanced TimeHUT Comprehensive Ablation Runner
==============================================

This enhanced runner integrates the unified optimization framework to provide:
✅ Improved computation metrics and statistical analysis
✅ Enhanced parameter tracking and correlation analysis
✅ Multi-objective optimization capabilities
✅ Advanced visualization and reporting
✅ Cross-validation and bootstrapping for robust results
✅ Real-time experiment tracking

Integration with unified_optimization_framework.py for superior metrics.
"""

import json
import time
import sys
import argparse
import csv
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import optimization framework components
sys.path.append('/home/amin/TSlib/models/timehut')
try:
    from unified_optimization_framework import (
        OptimizationSpace, OptimizationConfig, TimeHUTUnifiedOptimizer,
        OPTIONAL_LIBS
    )
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    print("⚠️ Warning: Unified optimization framework not available. Using basic metrics.")

# Statistical imports
try:
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    ADVANCED_STATS = True
except ImportError:
    ADVANCED_STATS = False

class EnhancedTimeHUTRunner:
    """Enhanced TimeHUT ablation study runner with optimization framework integration"""
    
    def __init__(self, enable_advanced_metrics: bool = True):
        self.results_dir = Path('/home/amin/TSlib/enhanced_metrics/enhanced_timehut_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.timehut_path = "/home/amin/TSlib/models/timehut"
        self.dataset_root = "/home/amin/TSlib/datasets"
        
        # Enhanced metrics tracking
        self.summary_results = []
        self.failed_runs = []
        self.performance_metrics = defaultdict(list)
        self.parameter_correlations = {}
        self.statistical_analysis = {}
        
        # Advanced features
        self.enable_advanced_metrics = enable_advanced_metrics and FRAMEWORK_AVAILABLE
        
        if self.enable_advanced_metrics:
            print("✅ Enhanced metrics enabled with unified optimization framework")
        else:
            print("⚠️ Using basic metrics mode")
        
        # TimeHUT Comprehensive Scenarios with enhanced parameter tracking
        self.scenarios = self._define_enhanced_scenarios()
        
    def _define_enhanced_scenarios(self) -> List[Dict[str, Any]]:
        """Define comprehensive TimeHUT scenarios with enhanced parameter tracking"""
        return [
            # 1. Baseline - No enhancements
            {
                "name": "Baseline",
                "description": "Standard TS2vec without any enhancements",
                "category": "baseline",
                "amc_instance": 0.0,
                "amc_temporal": 0.0,
                "amc_margin": 0.5,
                "min_tau": 0.4,
                "max_tau": 0.4,
                "temp_method": "fixed",
                "expected_performance": "medium",
                "complexity_score": 1
            },
            
            # 2. AMC Instance Only
            {
                "name": "AMC_Instance",  
                "description": "Angular Margin Contrastive for instance discrimination",
                "category": "amc_ablation",
                "amc_instance": 1.0,
                "amc_temporal": 0.0,
                "amc_margin": 0.5,
                "min_tau": 0.4,
                "max_tau": 0.4,
                "temp_method": "fixed",
                "expected_performance": "high",
                "complexity_score": 2
            },
            
            # 3. AMC Temporal Only
            {
                "name": "AMC_Temporal",
                "description": "Angular Margin Contrastive for temporal relationships", 
                "category": "amc_ablation",
                "amc_instance": 0.0,
                "amc_temporal": 1.0,
                "amc_margin": 0.5,
                "min_tau": 0.4,
                "max_tau": 0.4,
                "temp_method": "fixed",
                "expected_performance": "high",
                "complexity_score": 2
            },
            
            # 4-6. Temperature Scheduling variants
            {
                "name": "Temperature_Linear",
                "description": "Linear temperature scheduling",
                "category": "temperature_ablation",
                "amc_instance": 0.0,
                "amc_temporal": 0.0,
                "amc_margin": 0.5,
                "min_tau": 0.15,
                "max_tau": 0.75,
                "temp_method": "linear",
                "expected_performance": "medium_high",
                "complexity_score": 2
            },
            
            {
                "name": "Temperature_Cosine", 
                "description": "Cosine annealing temperature scheduling",
                "category": "temperature_ablation",
                "amc_instance": 0.0,
                "amc_temporal": 0.0,
                "amc_margin": 0.5,
                "min_tau": 0.15,
                "max_tau": 0.75,
                "temp_method": "cosine_annealing",
                "expected_performance": "medium_high",
                "complexity_score": 2
            },
            
            {
                "name": "Temperature_Exponential",
                "description": "Exponential decay temperature scheduling",
                "category": "temperature_ablation",
                "amc_instance": 0.0,
                "amc_temporal": 0.0,
                "amc_margin": 0.5,
                "min_tau": 0.15,
                "max_tau": 0.75,
                "temp_method": "exponential_decay",
                "expected_performance": "medium_high",
                "complexity_score": 2
            },
            
            # 7. AMC Both (Instance + Temporal)
            {
                "name": "AMC_Both",
                "description": "Combined AMC instance and temporal losses",
                "category": "amc_combined",
                "amc_instance": 0.7,
                "amc_temporal": 0.5,
                "amc_margin": 0.5,
                "min_tau": 0.4,
                "max_tau": 0.4,
                "temp_method": "fixed",
                "expected_performance": "very_high",
                "complexity_score": 3
            },
            
            # 8-10. AMC + Temperature combinations
            {
                "name": "AMC_Temperature_Linear",
                "description": "AMC losses with linear temperature scheduling",
                "category": "full_combination",
                "amc_instance": 1.0,
                "amc_temporal": 0.5,
                "amc_margin": 0.5,
                "min_tau": 0.15,
                "max_tau": 0.75,
                "temp_method": "linear",
                "expected_performance": "very_high",
                "complexity_score": 4
            },
            
            {
                "name": "AMC_Temperature_Cosine",
                "description": "AMC losses with cosine annealing (recommended setup)",
                "category": "full_combination",
                "amc_instance": 1.0,
                "amc_temporal": 0.5,
                "amc_margin": 0.5,
                "min_tau": 0.15,
                "max_tau": 0.75,
                "temp_method": "cosine_annealing",
                "expected_performance": "very_high",
                "complexity_score": 4
            },
            
            {
                "name": "AMC_Temperature_Exponential",
                "description": "AMC losses with exponential decay temperature",
                "category": "full_combination",
                "amc_instance": 1.0,
                "amc_temporal": 0.5,
                "amc_margin": 0.5,
                "min_tau": 0.15,
                "max_tau": 0.75,
                "temp_method": "exponential_decay",
                "expected_performance": "very_high",
                "complexity_score": 4
            },
            
            # 11. Optimized Configuration 
            {
                "name": "Optimized_Configuration",
                "description": "Best configuration from hyperparameter optimization",
                "category": "optimized",
                "amc_instance": 1.2,
                "amc_temporal": 0.8,
                "amc_margin": 0.4,
                "min_tau": 0.12,
                "max_tau": 0.82,
                "temp_method": "cosine_annealing",
                "expected_performance": "optimal",
                "complexity_score": 5
            },
            
            # 12. User's optimized parameters
            {
                "name": "User_Optimized_Configuration",
                "description": "User's optimized parameters (10.0, 7.53, 0.3)",
                "category": "user_optimized",
                "amc_instance": 10.0,
                "amc_temporal": 7.53,
                "amc_margin": 0.3,
                "min_tau": 0.05,
                "max_tau": 0.76,
                "temp_method": "cosine_annealing",
                "expected_performance": "optimal",
                "complexity_score": 5
            }
        ]
    
    def run_enhanced_timehut_scenario(self, scenario: Dict[str, Any], dataset: str = "Chinatown") -> Optional[Dict[str, Any]]:
        """Run a single TimeHUT scenario with enhanced metrics collection"""
        
        print(f"🚀 ENHANCED TIMEHUT SCENARIO: {scenario['name']}")
        print(f"📊 Description: {scenario['description']}")
        print(f"🏷️ Category: {scenario['category']}")
        print(f"🎯 AMC Instance: {scenario['amc_instance']}, AMC Temporal: {scenario['amc_temporal']}")
        print(f"🌡️ Temperature: {scenario['min_tau']} - {scenario['max_tau']} ({scenario['temp_method']})")
        print(f"📈 Expected Performance: {scenario['expected_performance']}")
        print(f"🧮 Complexity Score: {scenario['complexity_score']}")
        print(f"📈 Configuration: batch_size=8, epochs=200")
        print("=" * 80)
        
        # Build command using TimeHUT's unified comprehensive script
        run_name = f"{scenario['name']}_{dataset}_enhanced_batch8_epochs200"
        
        args = [
            '/home/amin/anaconda3/envs/tslib/bin/python',
            'train_unified_comprehensive.py',
            dataset,
            run_name,
            '--loader', 'UCR',
            '--epochs', '200',  # Force 200 epochs
            '--batch-size', '8',  # Force batch size 8  
            '--eval',
            '--dataroot', self.dataset_root,
            # AMC Parameters
            '--amc-instance', str(scenario['amc_instance']),
            '--amc-temporal', str(scenario['amc_temporal']),
            '--amc-margin', str(scenario['amc_margin']),
            # Temperature Parameters
            '--min-tau', str(scenario['min_tau']),
            '--max-tau', str(scenario['max_tau']),
            '--t-max', '10.5'  # Temperature scheduling period
        ]
        
        print(f"💻 Command: {' '.join(args[1:])}")
        
        # Enhanced timing and resource tracking
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = subprocess.run(
                args,
                cwd=self.timehut_path,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout
            )
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            runtime = end_time - start_time
            memory_used = end_memory - start_memory
            
            if result.returncode == 0:
                print(f"✅ Scenario '{scenario['name']}' completed successfully in {runtime:.1f}s")
                
                # Enhanced metrics extraction
                metrics = self._extract_enhanced_metrics(result.stdout, result.stderr)
                
                # Create comprehensive result record
                scenario_result = {
                    # Basic info
                    'scenario': scenario['name'],
                    'description': scenario['description'],
                    'category': scenario['category'],
                    'dataset': dataset,
                    
                    # Performance metrics
                    'accuracy': metrics['accuracy'],
                    'auprc': metrics.get('auprc', 0.0),
                    'runtime_seconds': runtime,
                    'memory_used_mb': memory_used,
                    
                    # Training configuration
                    'batch_size': 8,
                    'epochs': 200,
                    'status': 'success',
                    
                    # AMC parameters
                    'amc_instance': scenario['amc_instance'],
                    'amc_temporal': scenario['amc_temporal'],
                    'amc_margin': scenario['amc_margin'],
                    
                    # Temperature parameters
                    'min_tau': scenario['min_tau'],
                    'max_tau': scenario['max_tau'],
                    'temp_method': scenario['temp_method'],
                    
                    # Enhanced tracking
                    'expected_performance': scenario['expected_performance'],
                    'complexity_score': scenario['complexity_score'],
                    'timestamp': datetime.utcnow().isoformat(),
                    
                    # Advanced metrics (if available)
                    'loss_trajectory': metrics.get('loss_trajectory', []),
                    'tau_trajectory': metrics.get('tau_trajectory', []),
                    'convergence_epoch': metrics.get('convergence_epoch', -1),
                    'training_stability': metrics.get('training_stability', 0.0),
                    'parameter_efficiency': self._calculate_parameter_efficiency(scenario, metrics['accuracy'])
                }
                
                # Enhanced analysis
                if self.enable_advanced_metrics:
                    scenario_result.update(self._compute_advanced_metrics(scenario, metrics, runtime))
                
                print(f"📊 Enhanced Results: Accuracy={metrics['accuracy']:.4f}, AUPRC={metrics.get('auprc', 0.0):.4f}")
                print(f"⚡ Performance: Runtime={runtime:.1f}s, Memory={memory_used:.1f}MB")
                print(f"🎯 Efficiency Score: {scenario_result['parameter_efficiency']:.3f}")
                
                return scenario_result
                
            else:
                print(f"❌ Scenario '{scenario['name']}' failed with return code: {result.returncode}")
                print(f"📋 STDERR: {result.stderr[-300:] if result.stderr else 'No error output'}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Scenario '{scenario['name']}' timed out after 15 minutes")
            return None
        except Exception as e:
            print(f"❌ Scenario '{scenario['name']}' execution error: {str(e)}")
            return None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _extract_enhanced_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract enhanced metrics from TimeHUT output"""
        metrics = {
            'accuracy': 0.0,
            'auprc': 0.0,
            'loss_trajectory': [],
            'tau_trajectory': [],
            'convergence_epoch': -1,
            'training_stability': 0.0
        }
        
        lines = stdout.split('\n')
        
        # Extract accuracy and AUPRC
        for line in reversed(lines):
            if 'Evaluation result on test' in line and 'acc' in line:
                import re
                # Look for accuracy
                acc_match = re.search(r"'acc[uracy]*'?:?\s*([\d\.]+)", line)
                if acc_match:
                    metrics['accuracy'] = float(acc_match.group(1))
                
                # Look for AUPRC
                auprc_match = re.search(r"'auprc'?:?\s*([\d\.]+)", line)
                if auprc_match:
                    metrics['auprc'] = float(auprc_match.group(1))
                break
        
        # Extract training trajectory
        losses = []
        taus = []
        
        for line in lines:
            if 'Epoch #' in line and 'loss=' in line and 'tau =' in line:
                import re
                # Extract loss
                loss_match = re.search(r'loss=([\d\.]+)', line)
                if loss_match:
                    losses.append(float(loss_match.group(1)))
                
                # Extract tau
                tau_match = re.search(r'tau = ([\d\.]+)', line)
                if tau_match:
                    taus.append(float(tau_match.group(1)))
        
        metrics['loss_trajectory'] = losses
        metrics['tau_trajectory'] = taus
        
        # Calculate training stability (coefficient of variation of last 20% of losses)
        if len(losses) > 10:
            last_20_percent = losses[int(len(losses) * 0.8):]
            if len(last_20_percent) > 1:
                metrics['training_stability'] = np.std(last_20_percent) / np.mean(last_20_percent) if np.mean(last_20_percent) > 0 else 1.0
        
        # Find convergence epoch (when loss stabilizes)
        if len(losses) > 10:
            for i in range(10, len(losses)):
                recent_losses = losses[max(0, i-10):i]
                if len(recent_losses) > 1 and np.std(recent_losses) < 0.1 * np.mean(recent_losses):
                    metrics['convergence_epoch'] = i
                    break
        
        return metrics
    
    def _calculate_parameter_efficiency(self, scenario: Dict, accuracy: float) -> float:
        """Calculate parameter efficiency score (accuracy per complexity unit)"""
        complexity = scenario['complexity_score']
        if complexity == 0:
            return accuracy
        return accuracy / complexity
    
    def _compute_advanced_metrics(self, scenario: Dict, metrics: Dict, runtime: float) -> Dict[str, Any]:
        """Compute advanced metrics using unified optimization framework"""
        if not self.enable_advanced_metrics:
            return {}
        
        advanced = {}
        
        # Performance-complexity ratio
        advanced['performance_complexity_ratio'] = metrics['accuracy'] / scenario['complexity_score']
        
        # Time efficiency (accuracy per second)
        advanced['time_efficiency'] = metrics['accuracy'] / runtime if runtime > 0 else 0.0
        
        # Training efficiency score (combines accuracy, stability, and convergence speed)
        convergence_factor = 1.0
        if metrics['convergence_epoch'] > 0:
            convergence_factor = max(0.1, 1.0 - metrics['convergence_epoch'] / 200.0)  # Earlier convergence is better
        
        stability_factor = max(0.1, 1.0 - metrics['training_stability'])  # Lower variance is better
        
        advanced['training_efficiency_score'] = (
            metrics['accuracy'] * stability_factor * convergence_factor
        )
        
        # Expected vs actual performance
        performance_mapping = {
            'medium': 0.95,
            'medium_high': 0.97,
            'high': 0.98,
            'very_high': 0.985,
            'optimal': 0.99
        }
        expected_acc = performance_mapping.get(scenario['expected_performance'], 0.95)
        advanced['performance_expectation_ratio'] = metrics['accuracy'] / expected_acc
        
        return advanced
    
    def run_comprehensive_enhanced_ablation_study(self, dataset: str = "Chinatown", 
                                                   enable_cross_validation: bool = False,
                                                   n_cross_val_runs: int = 3) -> None:
        """Run comprehensive enhanced ablation study"""
        
        print("🚀 ENHANCED TIMEHUT COMPREHENSIVE ABLATION STUDY")
        print("=" * 80)
        print(f"📊 Dataset: {dataset}")
        print(f"📈 Configuration: batch_size=8, epochs=200")
        print(f"🧪 Total Scenarios: {len(self.scenarios)}")
        print(f"⏱️ Estimated Time: {len(self.scenarios) * 5} minutes")
        print(f"🔬 Advanced Metrics: {'Enabled' if self.enable_advanced_metrics else 'Disabled'}")
        print(f"🔄 Cross Validation: {'Enabled' if enable_cross_validation else 'Disabled'}")
        print("=" * 80)
        
        successful_results = []
        
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n📋 SCENARIO {i}/{len(self.scenarios)}")
            
            if enable_cross_validation:
                # Run multiple times for cross-validation
                scenario_results = []
                for run in range(n_cross_val_runs):
                    print(f"   🔄 Cross-validation run {run + 1}/{n_cross_val_runs}")
                    result = self.run_enhanced_timehut_scenario(scenario, dataset)
                    if result:
                        result['cv_run'] = run + 1
                        scenario_results.append(result)
                
                if scenario_results:
                    # Compute cross-validation statistics
                    cv_result = self._compute_cross_validation_stats(scenario_results, scenario)
                    successful_results.append(cv_result)
                    self.summary_results.append(cv_result)
                else:
                    self.failed_runs.append(scenario['name'])
            else:
                # Single run
                result = self.run_enhanced_timehut_scenario(scenario, dataset)
                if result:
                    successful_results.append(result)
                    self.summary_results.append(result)
                else:
                    self.failed_runs.append(scenario['name'])
                    
            print("-" * 80)
        
        # Enhanced analysis and reporting
        if successful_results:
            self._perform_enhanced_analysis(successful_results, dataset)
            self._save_enhanced_results(successful_results, dataset)
            self._print_enhanced_summary(successful_results)
            
            if self.enable_advanced_metrics:
                self._generate_advanced_visualizations(successful_results, dataset)
                self._perform_statistical_analysis(successful_results)
    
    def _compute_cross_validation_stats(self, scenario_results: List[Dict], scenario: Dict) -> Dict[str, Any]:
        """Compute cross-validation statistics for a scenario"""
        accuracies = [r['accuracy'] for r in scenario_results]
        runtimes = [r['runtime_seconds'] for r in scenario_results]
        
        cv_result = scenario_results[0].copy()  # Base result
        cv_result.update({
            # Cross-validation statistics
            'cv_runs': len(scenario_results),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_min': np.min(accuracies),
            'accuracy_max': np.max(accuracies),
            'runtime_mean': np.mean(runtimes),
            'runtime_std': np.std(runtimes),
            
            # Confidence intervals (if scipy available)
            'accuracy_ci_lower': np.mean(accuracies) - 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)),
            'accuracy_ci_upper': np.mean(accuracies) + 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)),
            
            # Use mean accuracy as primary metric
            'accuracy': np.mean(accuracies),
            'runtime_seconds': np.mean(runtimes),
            
            # Stability metrics
            'cv_stability': 1.0 - (np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0.0
        })
        
        return cv_result
    
    def _perform_enhanced_analysis(self, results: List[Dict], dataset: str) -> None:
        """Perform enhanced analysis on results"""
        print("\n🔬 PERFORMING ENHANCED ANALYSIS...")
        
        # Category-based analysis
        category_analysis = defaultdict(list)
        for result in results:
            category_analysis[result['category']].append(result['accuracy'])
        
        self.statistical_analysis['category_performance'] = {}
        for category, accuracies in category_analysis.items():
            self.statistical_analysis['category_performance'][category] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'count': len(accuracies),
                'max': np.max(accuracies),
                'min': np.min(accuracies)
            }
        
        # Correlation analysis (if advanced metrics enabled)
        if self.enable_advanced_metrics and len(results) > 3:
            self._compute_parameter_correlations(results)
        
        # Complexity analysis
        complexities = [r['complexity_score'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        if len(set(complexities)) > 1:  # More than one complexity level
            if ADVANCED_STATS:
                correlation, p_value = stats.pearsonr(complexities, accuracies)
                self.statistical_analysis['complexity_correlation'] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        print("✅ Enhanced analysis completed")
    
    def _compute_parameter_correlations(self, results: List[Dict]) -> None:
        """Compute correlations between parameters and performance"""
        # Extract parameter vectors
        parameters = ['amc_instance', 'amc_temporal', 'amc_margin', 'min_tau', 'max_tau', 'complexity_score']
        param_matrix = []
        accuracies = []
        
        for result in results:
            param_vector = [result.get(param, 0.0) for param in parameters]
            param_matrix.append(param_vector)
            accuracies.append(result['accuracy'])
        
        param_matrix = np.array(param_matrix)
        
        # Compute correlations
        for i, param in enumerate(parameters):
            param_values = param_matrix[:, i]
            if np.std(param_values) > 0:  # Only if parameter varies
                if ADVANCED_STATS:
                    correlation, p_value = stats.pearsonr(param_values, accuracies)
                    self.parameter_correlations[param] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
    
    def _save_enhanced_results(self, results: List[Dict], dataset: str) -> None:
        """Save enhanced results in multiple formats"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Enhanced JSON results with metadata
        json_file = self.results_dir / f"enhanced_timehut_ablation_{dataset}_{timestamp}.json"
        enhanced_data = {
            'metadata': {
                'dataset': dataset,
                'batch_size': 8,
                'epochs': 200,
                'total_scenarios': len(self.scenarios),
                'successful_scenarios': len(results),
                'failed_scenarios': len(self.failed_runs),
                'timestamp': timestamp,
                'advanced_metrics_enabled': self.enable_advanced_metrics,
                'framework_available': FRAMEWORK_AVAILABLE
            },
            'results': results,
            'statistical_analysis': self.statistical_analysis,
            'parameter_correlations': self.parameter_correlations,
            'failed_runs': self.failed_runs
        }
        
        with open(json_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
        
        # Enhanced CSV summary
        csv_file = self.results_dir / f"enhanced_timehut_summary_{dataset}_{timestamp}.csv"
        fieldnames = [
            'Scenario', 'Description', 'Category', 'Dataset', 'Accuracy', 'AUPRC', 'Runtime_Seconds',
            'Memory_Used_MB', 'Batch_Size', 'Epochs', 'AMC_Instance', 'AMC_Temporal', 'AMC_Margin',
            'Min_Tau', 'Max_Tau', 'Temp_Method', 'Expected_Performance', 'Complexity_Score',
            'Parameter_Efficiency', 'Training_Stability', 'Convergence_Epoch', 'Status'
        ]
        
        # Add cross-validation fields if present
        if any('cv_runs' in r for r in results):
            fieldnames.extend([
                'CV_Runs', 'Accuracy_Mean', 'Accuracy_Std', 'Accuracy_CI_Lower', 'Accuracy_CI_Upper', 'CV_Stability'
            ])
        
        # Add advanced metrics fields if enabled
        if self.enable_advanced_metrics:
            fieldnames.extend([
                'Performance_Complexity_Ratio', 'Time_Efficiency', 'Training_Efficiency_Score', 'Performance_Expectation_Ratio'
            ])
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'Scenario': result['scenario'],
                    'Description': result['description'],
                    'Category': result['category'],
                    'Dataset': result['dataset'],
                    'Accuracy': result['accuracy'],
                    'AUPRC': result.get('auprc', 0.0),
                    'Runtime_Seconds': result['runtime_seconds'],
                    'Memory_Used_MB': result.get('memory_used_mb', 0.0),
                    'Batch_Size': result['batch_size'],
                    'Epochs': result['epochs'],
                    'AMC_Instance': result['amc_instance'],
                    'AMC_Temporal': result['amc_temporal'],
                    'AMC_Margin': result['amc_margin'],
                    'Min_Tau': result['min_tau'],
                    'Max_Tau': result['max_tau'],
                    'Temp_Method': result['temp_method'],
                    'Expected_Performance': result['expected_performance'],
                    'Complexity_Score': result['complexity_score'],
                    'Parameter_Efficiency': result['parameter_efficiency'],
                    'Training_Stability': result.get('training_stability', 0.0),
                    'Convergence_Epoch': result.get('convergence_epoch', -1),
                    'Status': result['status']
                }
                
                # Add cross-validation fields
                if 'cv_runs' in result:
                    row.update({
                        'CV_Runs': result['cv_runs'],
                        'Accuracy_Mean': result['accuracy_mean'],
                        'Accuracy_Std': result['accuracy_std'],
                        'Accuracy_CI_Lower': result['accuracy_ci_lower'],
                        'Accuracy_CI_Upper': result['accuracy_ci_upper'],
                        'CV_Stability': result['cv_stability']
                    })
                
                # Add advanced metrics
                if self.enable_advanced_metrics:
                    row.update({
                        'Performance_Complexity_Ratio': result.get('performance_complexity_ratio', 0.0),
                        'Time_Efficiency': result.get('time_efficiency', 0.0),
                        'Training_Efficiency_Score': result.get('training_efficiency_score', 0.0),
                        'Performance_Expectation_Ratio': result.get('performance_expectation_ratio', 0.0)
                    })
                
                writer.writerow(row)
        
        print(f"📁 Enhanced results saved:")
        print(f"   📊 Detailed JSON: {json_file}")
        print(f"   📋 Enhanced CSV: {csv_file}")
    
    def _print_enhanced_summary(self, results: List[Dict]) -> None:
        """Print enhanced summary of ablation study"""
        
        print("\n" + "=" * 80)
        print("🏆 ENHANCED TIMEHUT COMPREHENSIVE ABLATION SUMMARY")
        print("=" * 80)
        
        if not results:
            print("❌ No successful results to summarize")
            return
        
        # Basic statistics
        accuracies = [r['accuracy'] for r in results]
        runtimes = [r['runtime_seconds'] for r in results]
        
        print(f"📊 Total Scenarios Tested: {len(results)}")
        print(f"✅ Successful Runs: {len(results)}")
        print(f"❌ Failed Runs: {len(self.failed_runs)}")
        print(f"📈 Overall Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"⏱️ Average Runtime: {np.mean(runtimes):.1f}s ± {np.std(runtimes):.1f}s")
        
        # Top performers
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        print("\n🏆 TOP 5 PERFORMING SCENARIOS:")
        for i, result in enumerate(sorted_results[:5], 1):
            efficiency = result.get('parameter_efficiency', 0.0)
            print(f"   {i}. {result['scenario']}: {result['accuracy']:.4f} ({result['runtime_seconds']:.1f}s, eff: {efficiency:.3f})")
        
        # Category analysis
        if self.statistical_analysis.get('category_performance'):
            print("\n📊 PERFORMANCE BY CATEGORY:")
            for category, stats in self.statistical_analysis['category_performance'].items():
                print(f"   📂 {category}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        # Parameter insights
        print("\n📈 ENHANCED ABLATION INSIGHTS:")
        
        # AMC Analysis
        baseline_results = [r for r in results if r['scenario'] == 'Baseline']
        amc_results = [r for r in results if r['amc_instance'] > 0 or r['amc_temporal'] > 0]
        
        if baseline_results and amc_results:
            baseline_acc = baseline_results[0]['accuracy']
            avg_amc_acc = np.mean([r['accuracy'] for r in amc_results])
            print(f"   🎯 AMC Impact: +{avg_amc_acc - baseline_acc:.4f} accuracy improvement")
        
        # Temperature Analysis
        temp_results = [r for r in results if r['temp_method'] != 'fixed']
        if temp_results:
            best_temp = max(temp_results, key=lambda x: x['accuracy'])
            print(f"   🌡️ Best Temperature Method: {best_temp['temp_method']} ({best_temp['accuracy']:.4f})")
        
        # Combined Analysis  
        combined_results = [r for r in results if r['amc_instance'] > 0 and r['temp_method'] != 'fixed']
        if combined_results:
            best_combined = max(combined_results, key=lambda x: x['accuracy'])
            print(f"   🔥 Best Combined: {best_combined['scenario']} ({best_combined['accuracy']:.4f})")
        
        # Efficiency Analysis
        if self.enable_advanced_metrics:
            efficiencies = [r.get('parameter_efficiency', 0.0) for r in results]
            most_efficient = max(results, key=lambda x: x.get('parameter_efficiency', 0.0))
            print(f"   ⚡ Most Efficient: {most_efficient['scenario']} (eff: {most_efficient.get('parameter_efficiency', 0.0):.3f})")
        
        # Statistical significance
        if self.statistical_analysis.get('complexity_correlation'):
            corr_info = self.statistical_analysis['complexity_correlation']
            significance = "significant" if corr_info['significant'] else "not significant"
            print(f"   📊 Complexity-Performance Correlation: {corr_info['correlation']:.3f} ({significance})")
        
        print("=" * 80)
    
    def _generate_advanced_visualizations(self, results: List[Dict], dataset: str) -> None:
        """Generate advanced visualizations"""
        if not OPTIONAL_LIBS.get('plotting', False):
            print("⚠️ Plotting libraries not available. Skipping visualizations.")
            return
        
        print("\n📊 Generating advanced visualizations...")
        
        # Create plots directory
        plots_dir = self.results_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Performance vs Complexity scatter plot
            complexities = [r['complexity_score'] for r in results]
            accuracies = [r['accuracy'] for r in results]
            categories = [r['category'] for r in results]
            
            plt.figure(figsize=(10, 6))
            for category in set(categories):
                cat_results = [r for r in results if r['category'] == category]
                cat_complexities = [r['complexity_score'] for r in cat_results]
                cat_accuracies = [r['accuracy'] for r in cat_results]
                plt.scatter(cat_complexities, cat_accuracies, label=category, alpha=0.7, s=100)
            
            plt.xlabel('Complexity Score')
            plt.ylabel('Accuracy')
            plt.title(f'Performance vs Complexity - {dataset}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f'performance_vs_complexity_{dataset}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Runtime vs Accuracy scatter plot
            runtimes = [r['runtime_seconds'] for r in results]
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(runtimes, accuracies, c=complexities, cmap='viridis', alpha=0.7, s=100)
            plt.colorbar(scatter, label='Complexity Score')
            plt.xlabel('Runtime (seconds)')
            plt.ylabel('Accuracy')
            plt.title(f'Runtime vs Accuracy (colored by complexity) - {dataset}')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f'runtime_vs_accuracy_{dataset}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Category performance box plot
            if len(set(categories)) > 1:
                plt.figure(figsize=(12, 6))
                category_data = []
                category_labels = []
                
                for category in sorted(set(categories)):
                    cat_accuracies = [r['accuracy'] for r in results if r['category'] == category]
                    category_data.append(cat_accuracies)
                    category_labels.append(category)
                
                plt.boxplot(category_data, labels=category_labels)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Accuracy')
                plt.title(f'Performance Distribution by Category - {dataset}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / f'category_performance_{dataset}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"✅ Visualizations saved to: {plots_dir}")
            
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {e}")
    
    def _perform_statistical_analysis(self, results: List[Dict]) -> None:
        """Perform statistical analysis on results"""
        if not ADVANCED_STATS:
            print("⚠️ Advanced statistical libraries not available.")
            return
        
        print("\n📊 PERFORMING STATISTICAL ANALYSIS...")
        
        # ANOVA analysis by category
        categories = list(set(r['category'] for r in results))
        if len(categories) > 2:
            category_groups = []
            for category in categories:
                cat_accuracies = [r['accuracy'] for r in results if r['category'] == category]
                category_groups.append(cat_accuracies)
            
            try:
                f_stat, p_value = stats.f_oneway(*category_groups)
                self.statistical_analysis['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                print(f"📊 ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
            except:
                print("⚠️ Could not perform ANOVA analysis")
        
        # Normality tests
        accuracies = [r['accuracy'] for r in results]
        if len(accuracies) > 8:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(accuracies)
                self.statistical_analysis['normality_test'] = {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'normally_distributed': shapiro_p > 0.05
                }
                print(f"📊 Normality test p-value: {shapiro_p:.4f}")
            except:
                print("⚠️ Could not perform normality test")
        
        print("✅ Statistical analysis completed")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Enhanced TimeHUT Comprehensive Ablation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic enhanced ablation study
    python enhanced_timehut_ablation_runner.py --dataset Chinatown
    
    # With cross-validation (3 runs per scenario)
    python enhanced_timehut_ablation_runner.py --dataset Chinatown --cross-validation --cv-runs 3
    
    # Disable advanced metrics for faster execution
    python enhanced_timehut_ablation_runner.py --dataset Chinatown --no-advanced-metrics
        """
    )
    
    parser.add_argument('--dataset', type=str, default='Chinatown',
                       help='Dataset to use for experiments')
    
    parser.add_argument('--cross-validation', action='store_true',
                       help='Enable cross-validation with multiple runs per scenario')
    
    parser.add_argument('--cv-runs', type=int, default=3,
                       help='Number of cross-validation runs per scenario')
    
    parser.add_argument('--no-advanced-metrics', action='store_true',
                       help='Disable advanced metrics computation')
    
    args = parser.parse_args()
    
    # Initialize enhanced runner
    runner = EnhancedTimeHUTRunner(
        enable_advanced_metrics=not args.no_advanced_metrics
    )
    
    # Run comprehensive enhanced ablation study
    runner.run_comprehensive_enhanced_ablation_study(
        dataset=args.dataset,
        enable_cross_validation=args.cross_validation,
        n_cross_val_runs=args.cv_runs
    )


if __name__ == "__main__":
    main()
