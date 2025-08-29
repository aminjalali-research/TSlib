#!/usr/bin/env python3
"""
TimeHUT Core Experiments Runner
==============================

Main experiment runner for comprehensive TimeHUT analysis.
Orchestrates all experiments including ablations, optimizations, and comparisons.

Author: TimeHUT Analysis Framework
Date: August 26, 2025
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timehut_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TimeHUTExperimentRunner:
    """Main experiment runner for TimeHUT analysis"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path("/home/amin/TSlib")
        self.unified_dir = self.base_dir / "unified"
        self.configs_dir = self.unified_dir / "timehut_configs"
        self.results_dir = self.unified_dir / "timehut_results"
        self.timehut_model_dir = self.base_dir / "models" / "timehut"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        for subdir in ["ablations", "optimizations", "comparisons", "visualizations", "reports"]:
            (self.results_dir / subdir).mkdir(exist_ok=True)
        
        # Load configurations
        self.configs = {}
        self._load_configs()
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        logger.info(f"🚀 TimeHUT Experiment Runner initialized")
        logger.info(f"   📁 Base Directory: {self.base_dir}")
        logger.info(f"   🆔 Experiment ID: {self.experiment_id}")
    
    def _load_configs(self):
        """Load all configuration files"""
        config_files = {
            'baseline': 'baseline_configs.json',
            'ablation': 'ablation_configs.json', 
            'optimization': 'optimization_configs.json',
            'cross_dataset': 'cross_dataset_configs.json'
        }
        
        for config_type, filename in config_files.items():
            config_path = self.configs_dir / filename
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.configs[config_type] = json.load(f)
                logger.info(f"✅ Loaded {config_type} configuration")
            else:
                logger.warning(f"⚠️  Configuration file not found: {config_path}")
    
    def run_single_experiment(self, 
                            dataset: str, 
                            scenario: str,
                            config: Dict[str, Any],
                            experiment_name: str = None) -> Dict[str, Any]:
        """Run a single TimeHUT experiment"""
        
        if experiment_name is None:
            experiment_name = f"{dataset}_{scenario}_{int(time.time())}"
        
        logger.info(f"🧪 Running experiment: {experiment_name}")
        
        # Build command
        cmd = [
            "python", 
            str(self.timehut_model_dir / "train_unified_comprehensive.py"),
            dataset,
            experiment_name,
            "--loader", "UCR"
        ]
        
        # Add scenario-specific parameters
        if 'scenario' in config:
            cmd.extend(["--scenario", config['scenario']])
        
        if 'epochs' in config:
            cmd.extend(["--epochs", str(config['epochs'])])
            
        if 'amc_instance' in config:
            cmd.extend(["--amc-instance", str(config['amc_instance'])])
            
        if 'amc_temporal' in config:
            cmd.extend(["--amc-temporal", str(config['amc_temporal'])])
            
        if 'amc_margin' in config:
            cmd.extend(["--amc-margin", str(config['amc_margin'])])
            
        if config.get('temperature_scheduling', False):
            if 'min_tau' in config:
                cmd.extend(["--min-tau", str(config['min_tau'])])
            if 'max_tau' in config:
                cmd.extend(["--max-tau", str(config['max_tau'])])
            if 't_max' in config:
                cmd.extend(["--t-max", str(config['t_max'])])
        
        if 'search_steps' in config:
            cmd.extend(["--search-steps", str(config['search_steps'])])
        
        # Execute experiment
        start_time = time.time()
        try:
            logger.info(f"   💻 Command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                cwd=str(self.base_dir),
                capture_output=True, 
                text=True, 
                timeout=config.get('timeout', 1800)  # 30 min default timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ Experiment completed: {experiment_name} ({execution_time:.2f}s)")
                
                # Parse results from output
                experiment_result = {
                    'experiment_name': experiment_name,
                    'dataset': dataset,
                    'scenario': scenario,
                    'config': config,
                    'execution_time': execution_time,
                    'status': 'success',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Try to extract metrics from output
                try:
                    experiment_result['metrics'] = self._parse_metrics(result.stdout)
                except Exception as e:
                    logger.warning(f"⚠️  Could not parse metrics: {e}")
                    experiment_result['metrics'] = {}
                
                return experiment_result
            else:
                logger.error(f"❌ Experiment failed: {experiment_name}")
                logger.error(f"   Error: {result.stderr}")
                return {
                    'experiment_name': experiment_name,
                    'dataset': dataset,
                    'scenario': scenario,
                    'config': config,
                    'execution_time': execution_time,
                    'status': 'failed',
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Experiment timed out: {experiment_name}")
            return {
                'experiment_name': experiment_name,
                'dataset': dataset,
                'scenario': scenario,
                'config': config,
                'status': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"💥 Experiment crashed: {experiment_name} - {e}")
            return {
                'experiment_name': experiment_name,
                'dataset': dataset,
                'scenario': scenario,
                'config': config,
                'status': 'crashed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_metrics(self, output: str) -> Dict[str, float]:
        """Parse metrics from experiment output"""
        metrics = {}
        
        # Common patterns to extract metrics
        patterns = {
            'accuracy': r'acc[uracy]*[:\s]+([0-9\.]+)',
            'auprc': r'auprc[:\s]+([0-9\.]+)',
            'f1': r'f1[:\s]+([0-9\.]+)',
            'training_time': r'training[_\s]time[:\s]+([0-9\.]+)',
            'memory_peak': r'memory[_\s]peak[:\s]+([0-9\.]+)',
        }
        
        import re
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric_name] = float(matches[-1])  # Take last match
                except ValueError:
                    pass
        
        return metrics
    
    def run_baseline_experiments(self) -> Dict[str, Any]:
        """Run baseline experiments"""
        logger.info("🏃 Running baseline experiments...")
        
        if 'baseline' not in self.configs:
            logger.error("❌ Baseline configuration not found")
            return {}
        
        baseline_config = self.configs['baseline']['baseline_experiments']
        results = {'baseline_experiments': []}
        
        for dataset in baseline_config['datasets']:
            for scenario_name, scenario_config in baseline_config['scenarios'].items():
                
                # Run multiple repeats
                for repeat in range(baseline_config.get('repeats', 1)):
                    experiment_name = f"baseline_{scenario_name}_{dataset}_rep{repeat}"
                    
                    config = {
                        'scenario': scenario_name,
                        'epochs': scenario_config['epochs'],
                        'amc_instance': scenario_config.get('amc_instance', 0.0),
                        'amc_temporal': scenario_config.get('amc_temporal', 0.0),
                        'amc_margin': scenario_config.get('amc_margin', 0.5),
                        'temperature_scheduling': scenario_config.get('temperature_scheduling', False)
                    }
                    
                    if scenario_config.get('temperature_scheduling', False):
                        config.update({
                            'min_tau': scenario_config.get('min_tau', 0.15),
                            'max_tau': scenario_config.get('max_tau', 0.75),
                            't_max': scenario_config.get('t_max', 10.5)
                        })
                    
                    result = self.run_single_experiment(dataset, scenario_name, config, experiment_name)
                    results['baseline_experiments'].append(result)
        
        # Save results
        output_file = self.results_dir / "comparisons" / f"baseline_results_{self.experiment_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Baseline experiments completed. Results saved to: {output_file}")
        return results
    
    def run_ablation_studies(self, ablation_type: str = 'all') -> Dict[str, Any]:
        """Run systematic ablation studies"""
        logger.info(f"🧪 Running ablation studies: {ablation_type}")
        
        if 'ablation' not in self.configs:
            logger.error("❌ Ablation configuration not found")
            return {}
        
        ablation_config = self.configs['ablation']['systematic_ablations']
        results = {'ablation_studies': []}
        
        if ablation_type in ['all', 'amc']:
            results.update(self._run_amc_ablations(ablation_config['amc_ablations']))
            
        if ablation_type in ['all', 'temperature']:
            results.update(self._run_temperature_ablations(ablation_config['temperature_ablations']))
            
        if ablation_type in ['all', 'combined']:
            results.update(self._run_combined_ablations(ablation_config['combined_ablations']))
        
        # Save results
        output_file = self.results_dir / "ablations" / f"ablation_results_{ablation_type}_{self.experiment_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Ablation studies completed. Results saved to: {output_file}")
        return results
    
    def _run_amc_ablations(self, amc_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run AMC parameter ablations"""
        logger.info("🔬 Running AMC ablations...")
        
        results = {'amc_ablations': []}
        param_grid = amc_config['parameter_grid']
        
        # Generate all parameter combinations
        import itertools
        param_combinations = list(itertools.product(
            param_grid['amc_instance'],
            param_grid['amc_temporal'], 
            param_grid['amc_margin']
        ))
        
        logger.info(f"   📊 Total AMC combinations: {len(param_combinations)}")
        
        for dataset in amc_config['datasets']:
            for amc_instance, amc_temporal, amc_margin in param_combinations:
                
                # Skip baseline (will be run separately)
                if amc_instance == 0.0 and amc_temporal == 0.0:
                    continue
                
                for repeat in range(amc_config.get('repeats', 1)):
                    experiment_name = f"amc_{dataset}_i{amc_instance}_t{amc_temporal}_m{amc_margin}_rep{repeat}"
                    
                    config = {
                        'scenario': 'amc_only',
                        'epochs': amc_config['epochs'],
                        'amc_instance': amc_instance,
                        'amc_temporal': amc_temporal,
                        'amc_margin': amc_margin,
                        'temperature_scheduling': False
                    }
                    
                    result = self.run_single_experiment(dataset, 'amc_ablation', config, experiment_name)
                    results['amc_ablations'].append(result)
        
        return results
    
    def _run_temperature_ablations(self, temp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run temperature scheduling ablations"""
        logger.info("🌡️  Running temperature ablations...")
        
        results = {'temperature_ablations': []}
        
        # Test different schedulers
        for scheduler_name, scheduler_config in temp_config['schedulers'].items():
            if not scheduler_config.get('enabled', True) and scheduler_name != 'none':
                continue
                
            for dataset in temp_config['datasets']:
                if scheduler_name == 'none':
                    # Baseline without temperature scheduling
                    config = {
                        'scenario': 'baseline',
                        'epochs': temp_config['epochs'],
                        'amc_instance': 0.0,
                        'amc_temporal': 0.0,
                        'temperature_scheduling': False
                    }
                else:
                    # With temperature scheduling
                    config = {
                        'scenario': 'temp_only',
                        'epochs': temp_config['epochs'],
                        'amc_instance': 0.0,
                        'amc_temporal': 0.0,
                        'temperature_scheduling': True,
                        'scheduler_method': scheduler_config['method'],
                        'min_tau': 0.15,
                        'max_tau': 0.75,
                        't_max': 10.5
                    }
                
                for repeat in range(temp_config.get('repeats', 1)):
                    experiment_name = f"temp_{scheduler_name}_{dataset}_rep{repeat}"
                    
                    result = self.run_single_experiment(dataset, 'temperature_ablation', config, experiment_name)
                    results['temperature_ablations'].append(result)
        
        return results
    
    def _run_combined_ablations(self, combined_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run combined AMC + temperature ablations"""
        logger.info("🔬🌡️ Running combined ablations...")
        
        results = {'combined_ablations': []}
        
        for scenario in combined_config['scenarios']:
            for dataset in combined_config['datasets']:
                for repeat in range(combined_config.get('repeats', 1)):
                    
                    experiment_name = f"combined_{scenario['name']}_{dataset}_rep{repeat}"
                    
                    config = {
                        'scenario': scenario['name'],
                        'epochs': combined_config['epochs'],
                        'amc_instance': scenario.get('amc_instance', 0.0),
                        'amc_temporal': scenario.get('amc_temporal', 0.0),
                        'temperature_scheduling': scenario.get('temperature_scheduling', False)
                    }
                    
                    if scenario.get('temperature_scheduling', False):
                        config.update({
                            'min_tau': scenario.get('min_tau', 0.15),
                            'max_tau': scenario.get('max_tau', 0.75),
                            't_max': scenario.get('t_max', 10.5)
                        })
                    
                    result = self.run_single_experiment(dataset, 'combined_ablation', config, experiment_name)
                    results['combined_ablations'].append(result)
        
        return results
    
    def run_optimization_studies(self, optimization_type: str = 'all') -> Dict[str, Any]:
        """Run hyperparameter optimization studies"""
        logger.info(f"⚙️ Running optimization studies: {optimization_type}")
        
        if 'optimization' not in self.configs:
            logger.error("❌ Optimization configuration not found")
            return {}
        
        optimization_config = self.configs['optimization']['hyperparameter_optimization']
        results = {'optimization_studies': []}
        
        if optimization_type in ['all', 'single']:
            results.update(self._run_single_objective_optimization(optimization_config['single_objective']))
            
        if optimization_type in ['all', 'multi']:
            results.update(self._run_multi_objective_optimization(optimization_config['multi_objective']))
        
        # Save results
        output_file = self.results_dir / "optimizations" / f"optimization_results_{optimization_type}_{self.experiment_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Optimization studies completed. Results saved to: {output_file}")
        return results
    
    def _run_single_objective_optimization(self, single_obj_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run single-objective optimization studies"""
        logger.info("🎯 Running single-objective optimization...")
        
        results = {'single_objective_optimization': []}
        
        for objective_name, objective_config in single_obj_config.items():
            logger.info(f"   🎯 Objective: {objective_name}")
            
            for dataset in objective_config['datasets']:
                experiment_name = f"opt_{objective_name}_{dataset}"
                
                config = {
                    'scenario': 'optimize_combined',
                    'epochs': objective_config['epochs'],
                    'search_steps': objective_config['search_steps'],
                    'optimization_objective': objective_config['objective']
                }
                
                result = self.run_single_experiment(dataset, 'single_objective_opt', config, experiment_name)
                results['single_objective_optimization'].append(result)
        
        return results
    
    def _run_multi_objective_optimization(self, multi_obj_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-objective optimization studies"""
        logger.info("🎯🎯 Running multi-objective optimization...")
        
        results = {'multi_objective_optimization': []}
        
        for objective_name, objective_config in multi_obj_config.items():
            logger.info(f"   🎯🎯 Objectives: {objective_name}")
            
            for dataset in objective_config['datasets']:
                experiment_name = f"multiopt_{objective_name}_{dataset}"
                
                config = {
                    'scenario': 'optimize_combined',
                    'epochs': objective_config['epochs'],
                    'search_steps': objective_config['search_steps'],
                    'optimization_objectives': objective_config['objectives'],
                    'objective_weights': objective_config['weights']
                }
                
                result = self.run_single_experiment(dataset, 'multi_objective_opt', config, experiment_name)
                results['multi_objective_optimization'].append(result)
        
        return results
    
    def run_cross_dataset_analysis(self) -> Dict[str, Any]:
        """Run cross-dataset analysis"""
        logger.info("🌐 Running cross-dataset analysis...")
        
        if 'cross_dataset' not in self.configs:
            logger.error("❌ Cross-dataset configuration not found")
            return {}
        
        cross_config = self.configs['cross_dataset']['cross_dataset_analysis']
        results = {'cross_dataset_analysis': []}
        
        # Run UCR dataset analysis
        results.update(self._run_ucr_analysis(cross_config['ucr_datasets']))
        
        # Run UEA dataset analysis
        results.update(self._run_uea_analysis(cross_config['uea_datasets']))
        
        # Save results
        output_file = self.results_dir / "comparisons" / f"cross_dataset_results_{self.experiment_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Cross-dataset analysis completed. Results saved to: {output_file}")
        return results
    
    def _run_ucr_analysis(self, ucr_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run UCR dataset analysis"""
        logger.info("📊 Running UCR dataset analysis...")
        
        results = {'ucr_analysis': []}
        
        for category_name, category_config in ucr_config.items():
            logger.info(f"   📊 Category: {category_name}")
            
            for dataset in category_config['datasets']:
                experiment_name = f"ucr_{category_name}_{dataset}"
                
                config = {
                    'scenario': 'optimize_combined',
                    'epochs': category_config['epochs'],
                    'search_steps': 20 if category_config['optimization_strategy'] == 'standard_search' else 10
                }
                
                result = self.run_single_experiment(dataset, 'ucr_analysis', config, experiment_name)
                result['category'] = category_name
                result['expected_behavior'] = category_config['expected_behavior']
                results['ucr_analysis'].append(result)
        
        return results
    
    def _run_uea_analysis(self, uea_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run UEA dataset analysis"""
        logger.info("📈 Running UEA dataset analysis...")
        
        results = {'uea_analysis': []}
        
        for analysis_type, analysis_config in uea_config.items():
            logger.info(f"   📈 Analysis: {analysis_type}")
            
            for dataset in analysis_config['datasets']:
                experiment_name = f"uea_{analysis_type}_{dataset}"
                
                config = {
                    'scenario': 'optimize_combined',
                    'epochs': analysis_config['epochs'],
                    'search_steps': 15,
                    'loader': 'UEA'  # Important for UEA datasets
                }
                
                result = self.run_single_experiment(dataset, 'uea_analysis', config, experiment_name)
                result['analysis_type'] = analysis_type
                results['uea_analysis'].append(result)
        
        return results
    
    def run_comprehensive_analysis(self, 
                                 experiment_types: List[str] = None,
                                 parallel: bool = False,
                                 max_workers: int = 4) -> Dict[str, Any]:
        """Run comprehensive analysis with all experiment types"""
        
        if experiment_types is None:
            experiment_types = ['baseline', 'ablation', 'optimization', 'cross_dataset']
        
        logger.info(f"🚀 Running comprehensive TimeHUT analysis")
        logger.info(f"   📋 Experiment types: {experiment_types}")
        logger.info(f"   ⚡ Parallel: {parallel}")
        
        all_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'experiment_types': experiment_types,
            'results': {}
        }
        
        if parallel:
            # Run experiments in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_experiment = {}
                
                if 'baseline' in experiment_types:
                    future = executor.submit(self.run_baseline_experiments)
                    future_to_experiment[future] = 'baseline'
                
                if 'ablation' in experiment_types:
                    future = executor.submit(self.run_ablation_studies, 'all')
                    future_to_experiment[future] = 'ablation'
                
                if 'optimization' in experiment_types:
                    future = executor.submit(self.run_optimization_studies, 'all')
                    future_to_experiment[future] = 'optimization'
                
                if 'cross_dataset' in experiment_types:
                    future = executor.submit(self.run_cross_dataset_analysis)
                    future_to_experiment[future] = 'cross_dataset'
                
                # Collect results as they complete
                for future in as_completed(future_to_experiment):
                    experiment_type = future_to_experiment[future]
                    try:
                        result = future.result()
                        all_results['results'][experiment_type] = result
                        logger.info(f"✅ Completed: {experiment_type}")
                    except Exception as e:
                        logger.error(f"❌ Failed {experiment_type}: {e}")
                        all_results['results'][experiment_type] = {'error': str(e)}
        else:
            # Run experiments sequentially
            if 'baseline' in experiment_types:
                all_results['results']['baseline'] = self.run_baseline_experiments()
            
            if 'ablation' in experiment_types:
                all_results['results']['ablation'] = self.run_ablation_studies('all')
            
            if 'optimization' in experiment_types:
                all_results['results']['optimization'] = self.run_optimization_studies('all')
            
            if 'cross_dataset' in experiment_types:
                all_results['results']['cross_dataset'] = self.run_cross_dataset_analysis()
        
        # Save comprehensive results
        output_file = self.results_dir / f"comprehensive_analysis_{self.experiment_id}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"🏁 Comprehensive analysis completed!")
        logger.info(f"   📊 Results saved to: {output_file}")
        
        return all_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TimeHUT Comprehensive Experiments Runner")
    
    parser.add_argument('--experiments', nargs='+', 
                       choices=['baseline', 'ablation', 'optimization', 'cross_dataset', 'all'],
                       default=['all'],
                       help='Experiment types to run')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    
    parser.add_argument('--base-dir', type=str, default='/home/amin/TSlib',
                       help='Base directory for TSlib')
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.experiments:
        args.experiments = ['baseline', 'ablation', 'optimization', 'cross_dataset']
    
    # Initialize runner
    runner = TimeHUTExperimentRunner(base_dir=args.base_dir)
    
    # Run experiments
    results = runner.run_comprehensive_analysis(
        experiment_types=args.experiments,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    
    print("\n" + "="*80)
    print("🏁 TIMEHUT COMPREHENSIVE ANALYSIS COMPLETED")
    print("="*80)
    print(f"📊 Experiment ID: {runner.experiment_id}")
    print(f"📁 Results Directory: {runner.results_dir}")
    print(f"🧪 Experiments Run: {', '.join(args.experiments)}")
    print("="*80)


if __name__ == "__main__":
    main()
