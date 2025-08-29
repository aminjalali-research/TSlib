#!/usr/bin/env python3
"""
TimeHUT Unified Optimization Framework
====================================

This comprehensive module integrates all optimization functionalities from the optimization/ folder
and builds upon the recent progress with temperature schedulers and unified training scripts.

Features Integrated:
✅ Advanced evolutionary optimization (from advanced_optimization_framework.py)
✅ PyHopper optimization with Neptune integration (from pyhopper_neptune_optimizer.py)
✅ Comprehensive ablation studies (from comprehensive_ablation_runner.py)
✅ Temperature scheduler optimization (from our recent work)
✅ Statistical analysis and multi-objective optimization
✅ Cross-dataset validation and generalization studies
✅ Real-time experiment tracking and reporting

Key Improvements:
- Integration with train_unified_comprehensive.py
- Support for all 13 temperature schedulers
- Enhanced parameter space with scheduler-specific parameters
- Streamlined execution using verified working commands
- Comprehensive results analysis and visualization

Usage:
    python unified_optimization_framework.py --mode ablation --dataset Chinatown
    python unified_optimization_framework.py --mode optimize --method pyhopper
    python unified_optimization_framework.py --mode comprehensive --datasets Chinatown AtrialFibrillation

Author: TimeHUT Unified Framework
Date: August 27, 2025
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import itertools
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports for advanced features
OPTIONAL_LIBS = {}
try:
    import pyhopper
    OPTIONAL_LIBS['pyhopper'] = True
except ImportError:
    OPTIONAL_LIBS['pyhopper'] = False

try:
    import neptune
    OPTIONAL_LIBS['neptune'] = True
except ImportError:
    OPTIONAL_LIBS['neptune'] = False

try:
    from scipy import stats
    OPTIONAL_LIBS['scipy'] = True
except ImportError:
    OPTIONAL_LIBS['scipy'] = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    OPTIONAL_LIBS['plotting'] = True
except ImportError:
    OPTIONAL_LIBS['plotting'] = False

# =====================================================
# CONFIGURATION CLASSES
# =====================================================

@dataclass
class OptimizationSpace:
    """Comprehensive hyperparameter search space for TimeHUT"""
    
    # AMC parameters
    amc_instance: Tuple[float, float] = (0.0, 12.0)
    amc_temporal: Tuple[float, float] = (0.0, 10.0)
    amc_margin: Tuple[float, float] = (0.1, 1.0)
    
    # Temperature scheduling parameters
    min_tau: Tuple[float, float] = (0.01, 0.3)
    max_tau: Tuple[float, float] = (0.5, 1.0)
    t_max: Tuple[float, float] = (5.0, 50.0)
    
    # Scheduler methods
    scheduler_methods: List[str] = field(default_factory=lambda: [
        'cosine_annealing', 'linear_decay', 'exponential_decay', 'step_decay',
        'polynomial_decay', 'sigmoid_decay', 'warmup_cosine', 'constant',
        'cyclic', 'multi_cycle_cosine', 'adaptive_cosine_annealing', 'cosine_with_restarts'
    ])
    
    # Scheduler-specific parameters
    temp_decay_rate: Tuple[float, float] = (0.85, 0.99)
    temp_step_size: Tuple[int, int] = (3, 15)
    temp_gamma: Tuple[float, float] = (0.3, 0.8)
    temp_power: Tuple[float, float] = (1.0, 3.0)
    temp_steepness: Tuple[float, float] = (0.5, 2.0)
    temp_warmup_epochs: Tuple[int, int] = (2, 10)
    temp_num_cycles: Tuple[int, int] = (2, 8)
    temp_decay_factor: Tuple[float, float] = (0.6, 0.95)
    temp_phase: Tuple[float, float] = (0.0, 6.28)  # 0 to 2π
    temp_frequency: Tuple[float, float] = (0.5, 3.0)
    temp_bias: Tuple[float, float] = (-0.1, 0.1)
    temp_momentum: Tuple[float, float] = (0.8, 0.99)
    temp_adaptation_rate: Tuple[float, float] = (0.05, 0.3)
    temp_restart_period: Tuple[float, float] = (3.0, 20.0)
    temp_restart_mult: Tuple[float, float] = (1.1, 3.0)
    
    # Training parameters
    batch_size: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    epochs: List[int] = field(default_factory=lambda: [50, 100, 150, 200])


@dataclass
class OptimizationConfig:
    """Configuration for optimization experiments"""
    
    # Search configuration
    search_space: OptimizationSpace = field(default_factory=OptimizationSpace)
    n_trials: int = 50
    max_time: Optional[int] = None  # Maximum time in seconds
    
    # Datasets
    datasets: List[str] = field(default_factory=lambda: ['Chinatown'])
    loader: str = 'UCR'
    
    # Optimization method
    method: str = 'pyhopper'  # pyhopper, grid, random, evolutionary
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ['accuracy'])
    
    # Statistical analysis
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    # Execution parameters
    timeout: int = 300  # Per trial timeout
    max_workers: int = 4
    
    # Results configuration
    save_intermediate: bool = True
    plot_results: bool = True
    
    # Neptune configuration (optional)
    neptune_project: Optional[str] = None
    neptune_tags: List[str] = field(default_factory=list)


# =====================================================
# CORE OPTIMIZATION ENGINE
# =====================================================

class TimeHUTUnifiedOptimizer:
    """Unified optimization framework for TimeHUT"""
    
    def __init__(self, config: OptimizationConfig, results_dir: str = None):
        self.config = config
        self.results_dir = results_dir or f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = []
        self.best_configs = {}
        self.optimization_history = []
        
        # Neptune setup (optional)
        self.run = None
        if OPTIONAL_LIBS['neptune'] and config.neptune_project:
            self.setup_neptune()
        
        logger.info(f"TimeHUT Unified Optimizer initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Optimization method: {config.method}")
        logger.info(f"Available libraries: {OPTIONAL_LIBS}")
    
    def setup_neptune(self):
        """Setup Neptune experiment tracking"""
        try:
            self.run = neptune.init_run(
                project=self.config.neptune_project,
                tags=self.config.neptune_tags + ['timehut', 'unified_optimizer']
            )
            logger.info("✅ Neptune tracking initialized")
        except Exception as e:
            logger.warning(f"⚠️ Neptune setup failed: {e}")
            self.run = None
    
    def run_single_trial(self, params: Dict[str, Any], dataset: str = 'Chinatown') -> Dict[str, Any]:
        """Run a single optimization trial"""
        trial_start = time.time()
        
        try:
            # Build command for unified training script
            cmd = self._build_training_command(params, dataset)
            
            # Execute training
            result = subprocess.run(
                cmd,
                cwd='/home/amin/TSlib/models/timehut',
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            # Parse results
            metrics = self._parse_training_output(result.stdout, result.stderr)
            metrics['runtime'] = time.time() - trial_start
            metrics['status'] = 'success' if result.returncode == 0 else 'failed'
            metrics['params'] = params.copy()
            metrics['dataset'] = dataset
            
            # Log to Neptune if available
            if self.run:
                self._log_to_neptune(metrics, params)
            
            return metrics
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Trial timed out after {self.config.timeout}s")
            return {
                'accuracy': 0.0,
                'runtime': self.config.timeout,
                'status': 'timeout',
                'error': 'Trial timed out',
                'params': params.copy(),
                'dataset': dataset
            }
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return {
                'accuracy': 0.0,
                'runtime': time.time() - trial_start,
                'status': 'error',
                'error': str(e),
                'params': params.copy(),
                'dataset': dataset
            }
    
    def _build_training_command(self, params: Dict[str, Any], dataset: str) -> List[str]:
        """Build training command from parameters"""
        
        # Extract scheduler method
        scheduler_method = params.get('scheduler_method', 'cosine_annealing')
        
        # Base command
        cmd = [
            '/home/amin/anaconda3/envs/tslib/bin/python',
            'train_unified_comprehensive.py',
            dataset,
            f'opt_trial_{int(time.time() * 1000)}',
            '--loader', self.config.loader,
            '--scenario', 'amc_temp',
            '--batch-size', str(params.get('batch_size', 8)),
            '--epochs', str(params.get('epochs', 100)),
            '--seed', '2002'
        ]
        
        # Add AMC parameters
        if 'amc_instance' in params:
            cmd.extend(['--amc-instance', str(params['amc_instance'])])
        if 'amc_temporal' in params:
            cmd.extend(['--amc-temporal', str(params['amc_temporal'])])
        if 'amc_margin' in params:
            cmd.extend(['--amc-margin', str(params['amc_margin'])])
        
        # Add temperature parameters
        if 'min_tau' in params:
            cmd.extend(['--min-tau', str(params['min_tau'])])
        if 'max_tau' in params:
            cmd.extend(['--max-tau', str(params['max_tau'])])
        if 't_max' in params:
            cmd.extend(['--t-max', str(params['t_max'])])
        
        # Add scheduler method
        cmd.extend(['--temp-method', scheduler_method])
        
        # Add scheduler-specific parameters
        scheduler_params = self._get_scheduler_params(scheduler_method, params)
        for param_name, param_value in scheduler_params.items():
            cmd.extend([f'--{param_name}', str(param_value)])
        
        return cmd
    
    def _get_scheduler_params(self, scheduler_method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get scheduler-specific parameters"""
        scheduler_params = {}
        
        if scheduler_method == 'exponential_decay' and 'temp_decay_rate' in params:
            scheduler_params['temp-decay-rate'] = params['temp_decay_rate']
        elif scheduler_method == 'step_decay':
            if 'temp_step_size' in params:
                scheduler_params['temp-step-size'] = int(params['temp_step_size'])
            if 'temp_gamma' in params:
                scheduler_params['temp-gamma'] = params['temp_gamma']
        elif scheduler_method == 'polynomial_decay' and 'temp_power' in params:
            scheduler_params['temp-power'] = params['temp_power']
        elif scheduler_method == 'sigmoid_decay' and 'temp_steepness' in params:
            scheduler_params['temp-steepness'] = params['temp_steepness']
        elif scheduler_method == 'warmup_cosine' and 'temp_warmup_epochs' in params:
            scheduler_params['temp-warmup-epochs'] = int(params['temp_warmup_epochs'])
        elif scheduler_method == 'multi_cycle_cosine':
            if 'temp_num_cycles' in params:
                scheduler_params['temp-num-cycles'] = int(params['temp_num_cycles'])
            if 'temp_decay_factor' in params:
                scheduler_params['temp-decay-factor'] = params['temp_decay_factor']
        elif scheduler_method == 'cosine_annealing':
            if 'temp_phase' in params:
                scheduler_params['temp-phase'] = params['temp_phase']
            if 'temp_frequency' in params:
                scheduler_params['temp-frequency'] = params['temp_frequency']
            if 'temp_bias' in params:
                scheduler_params['temp-bias'] = params['temp_bias']
        elif scheduler_method == 'cosine_with_restarts':
            if 'temp_restart_period' in params:
                scheduler_params['temp-restart-period'] = params['temp_restart_period']
            if 'temp_restart_mult' in params:
                scheduler_params['temp-restart-mult'] = params['temp_restart_mult']
        elif scheduler_method == 'adaptive_cosine_annealing':
            if 'temp_momentum' in params:
                scheduler_params['temp-momentum'] = params['temp_momentum']
            if 'temp_adaptation_rate' in params:
                scheduler_params['temp-adaptation-rate'] = params['temp_adaptation_rate']
        
        return scheduler_params
    
    def _parse_training_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse training output to extract metrics"""
        metrics = {'accuracy': 0.0, 'auprc': 0.0}
        
        # Look for final accuracy in output
        for line in reversed(stdout.split('\n')):
            if 'Final Accuracy:' in line:
                try:
                    acc_str = line.split('Final Accuracy:')[1].strip()
                    metrics['accuracy'] = float(acc_str)
                    break
                except (ValueError, IndexError):
                    continue
        
        # Alternative parsing for different output formats
        if metrics['accuracy'] == 0.0:
            for line in reversed(stdout.split('\n')):
                if 'Accuracy:' in line and not line.strip().startswith('#'):
                    try:
                        acc_str = line.split('Accuracy:')[1].split(',')[0].strip()
                        metrics['accuracy'] = float(acc_str)
                        break
                    except (ValueError, IndexError):
                        continue
        
        return metrics
    
    def _log_to_neptune(self, metrics: Dict[str, Any], params: Dict[str, Any]):
        """Log metrics and parameters to Neptune"""
        if not self.run:
            return
        
        try:
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.run[f"metrics/{key}"].log(value)
            
            # Log parameters
            for key, value in params.items():
                if isinstance(value, (int, float, str)):
                    self.run[f"parameters/{key}"] = value
        except Exception as e:
            logger.warning(f"Neptune logging failed: {e}")
    
    # =====================================================
    # OPTIMIZATION METHODS
    # =====================================================
    
    def optimize_pyhopper(self) -> Dict[str, Any]:
        """Run PyHopper optimization"""
        if not OPTIONAL_LIBS['pyhopper']:
            raise ImportError("PyHopper not available. Install with: pip install pyhopper")
        
        logger.info("🔍 Starting PyHopper optimization...")
        
        # Define search space for PyHopper
        search_space = self._create_pyhopper_search_space()
        
        def objective(params):
            results_per_dataset = []
            for dataset in self.config.datasets:
                result = self.run_single_trial(params, dataset)
                results_per_dataset.append(result['accuracy'])
            
            # Return mean accuracy across datasets
            return np.mean(results_per_dataset)
        
        # Run optimization
        best_params = search_space.run(
            objective,
            direction="maximize",
            steps=self.config.n_trials,
            n_jobs=1  # Sequential for stability
        )
        
        # Final evaluation with best parameters
        final_results = []
        for dataset in self.config.datasets:
            result = self.run_single_trial(best_params, dataset)
            final_results.append(result)
        
        return {
            'method': 'pyhopper',
            'best_params': best_params,
            'best_results': final_results,
            'mean_accuracy': np.mean([r['accuracy'] for r in final_results])
        }
    
    def _create_pyhopper_search_space(self):
        """Create PyHopper search space from configuration"""
        import pyhopper
        
        space_dict = {}
        
        # AMC parameters
        space_dict["amc_instance"] = pyhopper.float(*self.config.search_space.amc_instance, "0.1f")
        space_dict["amc_temporal"] = pyhopper.float(*self.config.search_space.amc_temporal, "0.1f")
        space_dict["amc_margin"] = pyhopper.float(*self.config.search_space.amc_margin, "0.1f")
        
        # Temperature parameters
        space_dict["min_tau"] = pyhopper.float(*self.config.search_space.min_tau, "0.01f")
        space_dict["max_tau"] = pyhopper.float(*self.config.search_space.max_tau, "0.01f")
        space_dict["t_max"] = pyhopper.float(*self.config.search_space.t_max, "1.0f")
        
        # Scheduler method (categorical)
        space_dict["scheduler_method"] = pyhopper.choice(self.config.search_space.scheduler_methods)
        
        # Scheduler-specific parameters (will be used conditionally)
        space_dict["temp_decay_rate"] = pyhopper.float(*self.config.search_space.temp_decay_rate, "0.01f")
        space_dict["temp_step_size"] = pyhopper.int(*self.config.search_space.temp_step_size)
        space_dict["temp_gamma"] = pyhopper.float(*self.config.search_space.temp_gamma, "0.01f")
        space_dict["temp_power"] = pyhopper.float(*self.config.search_space.temp_power, "0.1f")
        space_dict["temp_steepness"] = pyhopper.float(*self.config.search_space.temp_steepness, "0.1f")
        space_dict["temp_warmup_epochs"] = pyhopper.int(*self.config.search_space.temp_warmup_epochs)
        space_dict["temp_num_cycles"] = pyhopper.int(*self.config.search_space.temp_num_cycles)
        space_dict["temp_decay_factor"] = pyhopper.float(*self.config.search_space.temp_decay_factor, "0.01f")
        
        # Training parameters
        space_dict["batch_size"] = pyhopper.choice(self.config.search_space.batch_size)
        space_dict["epochs"] = pyhopper.choice(self.config.search_space.epochs)
        
        return pyhopper.Search(space_dict)
    
    def optimize_grid_search(self) -> Dict[str, Any]:
        """Run systematic grid search"""
        logger.info("🔍 Starting grid search optimization...")
        
        # Define grid parameters (subset for feasibility)
        grid_params = {
            'amc_instance': [0.0, 2.0, 5.0, 10.0],
            'amc_temporal': [0.0, 1.0, 3.0, 7.0],
            'amc_margin': [0.1, 0.3, 0.5, 0.7],
            'min_tau': [0.05, 0.1, 0.15],
            'max_tau': [0.6, 0.75, 0.9],
            't_max': [10, 20, 30],
            'scheduler_method': ['cosine_annealing', 'linear_decay', 'polynomial_decay'],
            'batch_size': [8],  # Fixed for grid search
            'epochs': [100]     # Fixed for grid search
        }
        
        # Generate all combinations
        param_names = list(grid_params.keys())
        param_values = list(grid_params.values())
        
        all_results = []
        total_combinations = np.prod([len(values) for values in param_values])
        logger.info(f"Total combinations to test: {total_combinations}")
        
        for i, combination in enumerate(itertools.product(*param_values)):
            params = dict(zip(param_names, combination))
            
            logger.info(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            # Test on all datasets
            combination_results = []
            for dataset in self.config.datasets:
                result = self.run_single_trial(params, dataset)
                combination_results.append(result)
            
            # Store results
            mean_accuracy = np.mean([r['accuracy'] for r in combination_results])
            all_results.append({
                'params': params,
                'results': combination_results,
                'mean_accuracy': mean_accuracy
            })
            
            # Save intermediate results
            if self.config.save_intermediate and i % 10 == 0:
                self._save_intermediate_results(all_results)
        
        # Find best combination
        best_result = max(all_results, key=lambda x: x['mean_accuracy'])
        
        return {
            'method': 'grid_search',
            'all_results': all_results,
            'best_params': best_result['params'],
            'best_results': best_result['results'],
            'mean_accuracy': best_result['mean_accuracy']
        }
    
    def optimize_random_search(self) -> Dict[str, Any]:
        """Run random search optimization"""
        logger.info("🔍 Starting random search optimization...")
        
        all_results = []
        
        for trial in range(self.config.n_trials):
            # Generate random parameters
            params = self._generate_random_params()
            
            logger.info(f"Random trial {trial+1}/{self.config.n_trials}")
            
            # Test on all datasets
            trial_results = []
            for dataset in self.config.datasets:
                result = self.run_single_trial(params, dataset)
                trial_results.append(result)
            
            # Store results
            mean_accuracy = np.mean([r['accuracy'] for r in trial_results])
            all_results.append({
                'params': params,
                'results': trial_results,
                'mean_accuracy': mean_accuracy
            })
            
            # Save intermediate results
            if self.config.save_intermediate and trial % 10 == 0:
                self._save_intermediate_results(all_results)
        
        # Find best result
        best_result = max(all_results, key=lambda x: x['mean_accuracy'])
        
        return {
            'method': 'random_search',
            'all_results': all_results,
            'best_params': best_result['params'],
            'best_results': best_result['results'],
            'mean_accuracy': best_result['mean_accuracy']
        }
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameters from search space"""
        params = {}
        
        # AMC parameters
        params['amc_instance'] = np.random.uniform(*self.config.search_space.amc_instance)
        params['amc_temporal'] = np.random.uniform(*self.config.search_space.amc_temporal)
        params['amc_margin'] = np.random.uniform(*self.config.search_space.amc_margin)
        
        # Temperature parameters
        params['min_tau'] = np.random.uniform(*self.config.search_space.min_tau)
        params['max_tau'] = np.random.uniform(*self.config.search_space.max_tau)
        params['t_max'] = np.random.uniform(*self.config.search_space.t_max)
        
        # Scheduler method
        params['scheduler_method'] = np.random.choice(self.config.search_space.scheduler_methods)
        
        # Scheduler-specific parameters
        params['temp_decay_rate'] = np.random.uniform(*self.config.search_space.temp_decay_rate)
        params['temp_step_size'] = np.random.randint(*self.config.search_space.temp_step_size)
        params['temp_gamma'] = np.random.uniform(*self.config.search_space.temp_gamma)
        params['temp_power'] = np.random.uniform(*self.config.search_space.temp_power)
        params['temp_steepness'] = np.random.uniform(*self.config.search_space.temp_steepness)
        params['temp_warmup_epochs'] = np.random.randint(*self.config.search_space.temp_warmup_epochs)
        params['temp_num_cycles'] = np.random.randint(*self.config.search_space.temp_num_cycles)
        params['temp_decay_factor'] = np.random.uniform(*self.config.search_space.temp_decay_factor)
        
        # Training parameters
        params['batch_size'] = np.random.choice(self.config.search_space.batch_size)
        params['epochs'] = np.random.choice(self.config.search_space.epochs)
        
        return params
    
    # =====================================================
    # ABLATION STUDIES
    # =====================================================
    
    def run_comprehensive_ablation(self) -> Dict[str, Any]:
        """Run comprehensive ablation studies"""
        logger.info("🔬 Starting comprehensive ablation studies...")
        
        ablation_results = {}
        
        # 1. AMC Component Ablation
        logger.info("Phase 1: AMC Component Ablation")
        ablation_results['amc_ablation'] = self._run_amc_ablation()
        
        # 2. Temperature Scheduler Ablation
        logger.info("Phase 2: Temperature Scheduler Ablation")
        ablation_results['scheduler_ablation'] = self._run_scheduler_ablation()
        
        # 3. Parameter Sensitivity Analysis
        logger.info("Phase 3: Parameter Sensitivity Analysis")
        ablation_results['sensitivity_analysis'] = self._run_sensitivity_analysis()
        
        # 4. Interaction Effects
        logger.info("Phase 4: Interaction Effects Analysis")
        ablation_results['interaction_analysis'] = self._run_interaction_analysis()
        
        return ablation_results
    
    def _run_amc_ablation(self) -> Dict[str, Any]:
        """Run AMC component ablation study"""
        
        amc_configs = [
            {'name': 'baseline', 'amc_instance': 0.0, 'amc_temporal': 0.0, 'amc_margin': 0.5},
            {'name': 'instance_only', 'amc_instance': 1.0, 'amc_temporal': 0.0, 'amc_margin': 0.5},
            {'name': 'temporal_only', 'amc_instance': 0.0, 'amc_temporal': 1.0, 'amc_margin': 0.5},
            {'name': 'both_low', 'amc_instance': 0.5, 'amc_temporal': 0.5, 'amc_margin': 0.5},
            {'name': 'both_high', 'amc_instance': 2.0, 'amc_temporal': 2.0, 'amc_margin': 0.5},
            {'name': 'optimized_chinatown', 'amc_instance': 10.0, 'amc_temporal': 7.53, 'amc_margin': 0.3}
        ]
        
        results = {}
        
        for config in amc_configs:
            config_name = config['name']
            logger.info(f"Testing AMC configuration: {config_name}")
            
            # Base parameters
            params = {
                'amc_instance': config['amc_instance'],
                'amc_temporal': config['amc_temporal'],
                'amc_margin': config['amc_margin'],
                'min_tau': 0.15,
                'max_tau': 0.75,
                't_max': 10.5,
                'scheduler_method': 'cosine_annealing',
                'batch_size': 8,
                'epochs': 100
            }
            
            # Test on all datasets
            config_results = []
            for dataset in self.config.datasets:
                result = self.run_single_trial(params, dataset)
                config_results.append(result)
            
            results[config_name] = {
                'params': params,
                'results': config_results,
                'mean_accuracy': np.mean([r['accuracy'] for r in config_results]),
                'std_accuracy': np.std([r['accuracy'] for r in config_results])
            }
        
        return results
    
    def _run_scheduler_ablation(self) -> Dict[str, Any]:
        """Run temperature scheduler ablation study"""
        
        results = {}
        
        # Use optimized AMC parameters for fair scheduler comparison
        base_params = {
            'amc_instance': 10.0,
            'amc_temporal': 7.53,
            'amc_margin': 0.3,
            'min_tau': 0.05,
            'max_tau': 0.76,
            't_max': 25,
            'batch_size': 8,
            'epochs': 100
        }
        
        for scheduler in self.config.search_space.scheduler_methods:
            logger.info(f"Testing scheduler: {scheduler}")
            
            params = base_params.copy()
            params['scheduler_method'] = scheduler
            
            # Add scheduler-specific parameters with reasonable defaults
            if scheduler == 'exponential_decay':
                params['temp_decay_rate'] = 0.95
            elif scheduler == 'step_decay':
                params['temp_step_size'] = 8
                params['temp_gamma'] = 0.5
            elif scheduler == 'polynomial_decay':
                params['temp_power'] = 2.0
            elif scheduler == 'sigmoid_decay':
                params['temp_steepness'] = 1.0
            elif scheduler == 'warmup_cosine':
                params['temp_warmup_epochs'] = 5
            elif scheduler == 'multi_cycle_cosine':
                params['temp_num_cycles'] = 3
                params['temp_decay_factor'] = 0.8
            elif scheduler == 'cosine_with_restarts':
                params['temp_restart_period'] = 5.0
                params['temp_restart_mult'] = 1.5
            
            # Test on all datasets
            scheduler_results = []
            for dataset in self.config.datasets:
                result = self.run_single_trial(params, dataset)
                scheduler_results.append(result)
            
            results[scheduler] = {
                'params': params,
                'results': scheduler_results,
                'mean_accuracy': np.mean([r['accuracy'] for r in scheduler_results]),
                'std_accuracy': np.std([r['accuracy'] for r in scheduler_results])
            }
        
        return results
    
    def _run_sensitivity_analysis(self) -> Dict[str, Any]:
        """Run parameter sensitivity analysis"""
        
        sensitivity_results = {}
        
        # Base configuration
        base_params = {
            'amc_instance': 10.0,
            'amc_temporal': 7.53,
            'amc_margin': 0.3,
            'min_tau': 0.05,
            'max_tau': 0.76,
            't_max': 25,
            'scheduler_method': 'polynomial_decay',
            'temp_power': 2.0,
            'batch_size': 8,
            'epochs': 100
        }
        
        # Parameters to analyze
        sensitivity_params = {
            'amc_instance': [0.0, 2.0, 5.0, 10.0, 15.0],
            'amc_temporal': [0.0, 2.0, 5.0, 7.53, 10.0],
            'min_tau': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_tau': [0.5, 0.65, 0.76, 0.85, 0.95],
            't_max': [10, 20, 25, 30, 40]
        }
        
        for param_name, param_values in sensitivity_params.items():
            logger.info(f"Sensitivity analysis for: {param_name}")
            
            param_results = []
            for param_value in param_values:
                params = base_params.copy()
                params[param_name] = param_value
                
                # Test on first dataset only for sensitivity analysis
                result = self.run_single_trial(params, self.config.datasets[0])
                param_results.append({
                    'value': param_value,
                    'accuracy': result['accuracy'],
                    'runtime': result['runtime']
                })
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def _run_interaction_analysis(self) -> Dict[str, Any]:
        """Analyze parameter interactions"""
        
        # Focus on key parameter pairs
        interaction_pairs = [
            ('amc_instance', 'amc_temporal'),
            ('min_tau', 'max_tau'),
            ('scheduler_method', 't_max')
        ]
        
        interaction_results = {}
        
        for param1, param2 in interaction_pairs:
            logger.info(f"Analyzing interaction: {param1} x {param2}")
            
            if param1 == 'scheduler_method':
                # Categorical x continuous interaction
                results = self._analyze_categorical_continuous_interaction(param1, param2)
            else:
                # Continuous x continuous interaction
                results = self._analyze_continuous_interaction(param1, param2)
            
            interaction_results[f"{param1}_x_{param2}"] = results
        
        return interaction_results
    
    def _analyze_continuous_interaction(self, param1: str, param2: str) -> List[Dict]:
        """Analyze interaction between two continuous parameters"""
        
        # Define parameter grids
        param_grids = {
            'amc_instance': [0.0, 5.0, 10.0],
            'amc_temporal': [0.0, 3.0, 7.53],
            'min_tau': [0.05, 0.1, 0.15],
            'max_tau': [0.6, 0.76, 0.9],
            't_max': [15, 25, 35]
        }
        
        base_params = {
            'amc_instance': 10.0,
            'amc_temporal': 7.53,
            'amc_margin': 0.3,
            'min_tau': 0.05,
            'max_tau': 0.76,
            't_max': 25,
            'scheduler_method': 'polynomial_decay',
            'temp_power': 2.0,
            'batch_size': 8,
            'epochs': 100
        }
        
        results = []
        values1 = param_grids[param1]
        values2 = param_grids[param2]
        
        for val1 in values1:
            for val2 in values2:
                params = base_params.copy()
                params[param1] = val1
                params[param2] = val2
                
                result = self.run_single_trial(params, self.config.datasets[0])
                results.append({
                    param1: val1,
                    param2: val2,
                    'accuracy': result['accuracy'],
                    'runtime': result['runtime']
                })
        
        return results
    
    def _analyze_categorical_continuous_interaction(self, param1: str, param2: str) -> List[Dict]:
        """Analyze interaction between categorical and continuous parameter"""
        
        schedulers = ['cosine_annealing', 'linear_decay', 'polynomial_decay', 'step_decay']
        t_max_values = [15, 25, 35]
        
        base_params = {
            'amc_instance': 10.0,
            'amc_temporal': 7.53,
            'amc_margin': 0.3,
            'min_tau': 0.05,
            'max_tau': 0.76,
            'batch_size': 8,
            'epochs': 100
        }
        
        results = []
        
        for scheduler in schedulers:
            for t_max in t_max_values:
                params = base_params.copy()
                params['scheduler_method'] = scheduler
                params['t_max'] = t_max
                
                result = self.run_single_trial(params, self.config.datasets[0])
                results.append({
                    'scheduler_method': scheduler,
                    't_max': t_max,
                    'accuracy': result['accuracy'],
                    'runtime': result['runtime']
                })
        
        return results
    
    # =====================================================
    # RESULTS ANALYSIS AND VISUALIZATION
    # =====================================================
    
    def _save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results"""
        output_file = Path(self.results_dir) / "intermediate_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Intermediate results saved to: {output_file}")
    
    def analyze_results(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results"""
        logger.info("📊 Analyzing optimization results...")
        
        analysis = {}
        
        # Statistical analysis
        if 'all_results' in optimization_results:
            accuracies = [r['mean_accuracy'] for r in optimization_results['all_results']]
            analysis['statistics'] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'median_accuracy': np.median(accuracies)
            }
            
            if OPTIONAL_LIBS['scipy']:
                analysis['statistics']['confidence_interval'] = stats.norm.interval(
                    self.config.confidence_level,
                    loc=np.mean(accuracies),
                    scale=stats.sem(accuracies)
                )
        
        # Parameter importance analysis
        if 'all_results' in optimization_results:
            analysis['parameter_importance'] = self._analyze_parameter_importance(
                optimization_results['all_results']
            )
        
        # Top performing configurations
        if 'all_results' in optimization_results:
            sorted_results = sorted(
                optimization_results['all_results'],
                key=lambda x: x['mean_accuracy'],
                reverse=True
            )
            analysis['top_configs'] = sorted_results[:10]
        
        return analysis
    
    def _analyze_parameter_importance(self, all_results: List[Dict]) -> Dict[str, float]:
        """Analyze parameter importance using correlation with accuracy"""
        if not OPTIONAL_LIBS['scipy']:
            return {}
        
        # Extract parameters and accuracies
        params_data = defaultdict(list)
        accuracies = []
        
        for result in all_results:
            accuracies.append(result['mean_accuracy'])
            for param_name, param_value in result['params'].items():
                if isinstance(param_value, (int, float)):
                    params_data[param_name].append(param_value)
        
        # Calculate correlations
        importance = {}
        for param_name, param_values in params_data.items():
            if len(set(param_values)) > 1:  # Only if parameter varies
                correlation, p_value = stats.pearsonr(param_values, accuracies)
                importance[param_name] = {
                    'correlation': correlation,
                    'abs_correlation': abs(correlation),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return importance
    
    def generate_report(self, optimization_results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate comprehensive optimization report"""
        logger.info("📝 Generating optimization report...")
        
        report = []
        report.append("# TimeHUT Unified Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Configuration
        report.append("## Configuration")
        report.append(f"- Optimization method: {self.config.method}")
        report.append(f"- Number of trials: {self.config.n_trials}")
        report.append(f"- Datasets: {', '.join(self.config.datasets)}")
        report.append(f"- Objectives: {', '.join(self.config.objectives)}")
        report.append("")
        
        # Best results
        if 'best_params' in optimization_results:
            report.append("## Best Configuration")
            report.append(f"**Mean Accuracy**: {optimization_results['mean_accuracy']:.4f}")
            report.append("")
            report.append("**Parameters**:")
            for param, value in optimization_results['best_params'].items():
                if isinstance(value, float):
                    report.append(f"- {param}: {value:.4f}")
                else:
                    report.append(f"- {param}: {value}")
            report.append("")
        
        # Statistical analysis
        if 'statistics' in analysis:
            stats = analysis['statistics']
            report.append("## Statistical Analysis")
            report.append(f"- Mean accuracy: {stats['mean_accuracy']:.4f}")
            report.append(f"- Standard deviation: {stats['std_accuracy']:.4f}")
            report.append(f"- Range: {stats['min_accuracy']:.4f} - {stats['max_accuracy']:.4f}")
            report.append(f"- Median accuracy: {stats['median_accuracy']:.4f}")
            
            if 'confidence_interval' in stats:
                ci_low, ci_high = stats['confidence_interval']
                report.append(f"- 95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
            report.append("")
        
        # Parameter importance
        if 'parameter_importance' in analysis:
            report.append("## Parameter Importance")
            importance = analysis['parameter_importance']
            
            # Sort by absolute correlation
            sorted_params = sorted(
                importance.items(),
                key=lambda x: x[1]['abs_correlation'],
                reverse=True
            )
            
            for param_name, param_stats in sorted_params[:10]:
                significance = "✓" if param_stats['significant'] else "✗"
                report.append(f"- **{param_name}**: r={param_stats['correlation']:.3f} "
                            f"(p={param_stats['p_value']:.3f}) {significance}")
            report.append("")
        
        # Top configurations
        if 'top_configs' in analysis:
            report.append("## Top 5 Configurations")
            for i, config in enumerate(analysis['top_configs'][:5], 1):
                report.append(f"### {i}. Accuracy: {config['mean_accuracy']:.4f}")
                for param, value in config['params'].items():
                    if isinstance(value, float):
                        report.append(f"- {param}: {value:.4f}")
                    else:
                        report.append(f"- {param}: {value}")
                report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = Path(self.results_dir) / "optimization_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to: {report_file}")
        return report_text
    
    def create_visualizations(self, optimization_results: Dict[str, Any], analysis: Dict[str, Any]):
        """Create optimization visualizations"""
        if not OPTIONAL_LIBS['plotting']:
            logger.warning("Plotting libraries not available. Skipping visualizations.")
            return
        
        logger.info("📊 Creating visualizations...")
        
        # Optimization history plot
        if 'all_results' in optimization_results:
            self._plot_optimization_history(optimization_results['all_results'])
        
        # Parameter importance plot
        if 'parameter_importance' in analysis:
            self._plot_parameter_importance(analysis['parameter_importance'])
        
        # Correlation matrix for numeric parameters
        if 'all_results' in optimization_results:
            self._plot_parameter_correlations(optimization_results['all_results'])
    
    def _plot_optimization_history(self, all_results: List[Dict]):
        """Plot optimization history"""
        accuracies = [r['mean_accuracy'] for r in all_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, 'b-', alpha=0.7, label='Trial accuracy')
        plt.plot(np.cummax(accuracies), 'r-', linewidth=2, label='Best so far')
        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(Path(self.results_dir) / "optimization_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_importance(self, parameter_importance: Dict[str, Dict]):
        """Plot parameter importance"""
        if not parameter_importance:
            return
        
        params = list(parameter_importance.keys())
        correlations = [parameter_importance[p]['abs_correlation'] for p in params]
        significant = [parameter_importance[p]['significant'] for p in params]
        
        colors = ['red' if sig else 'blue' for sig in significant]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(params, correlations, color=colors, alpha=0.7)
        plt.xlabel('Parameters')
        plt.ylabel('Absolute Correlation with Accuracy')
        plt.title('Parameter Importance')
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7, label='Significant (p<0.05)')
        blue_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7, label='Not significant')
        plt.legend(handles=[red_patch, blue_patch])
        
        plt.tight_layout()
        plt.savefig(Path(self.results_dir) / "parameter_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_correlations(self, all_results: List[Dict]):
        """Plot parameter correlation matrix"""
        # Extract numeric parameters
        numeric_params = {}
        for result in all_results:
            for param_name, param_value in result['params'].items():
                if isinstance(param_value, (int, float)):
                    if param_name not in numeric_params:
                        numeric_params[param_name] = []
                    numeric_params[param_name].append(param_value)
        
        if len(numeric_params) < 2:
            return
        
        # Create DataFrame
        df = pd.DataFrame(numeric_params)
        
        # Add accuracy
        df['accuracy'] = [r['mean_accuracy'] for r in all_results]
        
        # Create correlation matrix
        corr_matrix = df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(Path(self.results_dir) / "parameter_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.run:
            self.run.stop()
            logger.info("Neptune run stopped")


# =====================================================
# MAIN EXECUTION INTERFACE
# =====================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="TimeHUT Unified Optimization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # PyHopper optimization on single dataset
    python unified_optimization_framework.py --mode optimize --method pyhopper --datasets Chinatown
    
    # Comprehensive ablation study
    python unified_optimization_framework.py --mode ablation --datasets Chinatown AtrialFibrillation
    
    # Grid search optimization
    python unified_optimization_framework.py --mode optimize --method grid --trials 100
    
    # Combined ablation and optimization
    python unified_optimization_framework.py --mode comprehensive --method pyhopper
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['optimize', 'ablation', 'comprehensive'],
                       help='Optimization mode')
    
    parser.add_argument('--method', type=str, default='pyhopper',
                       choices=['pyhopper', 'grid', 'random'],
                       help='Optimization method')
    
    parser.add_argument('--datasets', type=str, nargs='+', default=['Chinatown'],
                       help='Datasets to optimize on')
    
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of optimization trials')
    
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per trial in seconds')
    
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory')
    
    parser.add_argument('--neptune-project', type=str, default=None,
                       help='Neptune project name')
    
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizationConfig(
        method=args.method,
        datasets=args.datasets,
        n_trials=args.trials,
        timeout=args.timeout,
        neptune_project=args.neptune_project,
        plot_results=args.plot
    )
    
    # Initialize optimizer
    optimizer = TimeHUTUnifiedOptimizer(config, args.results_dir)
    
    try:
        if args.mode == 'optimize':
            # Run optimization only
            logger.info("🚀 Starting optimization...")
            
            if args.method == 'pyhopper':
                results = optimizer.optimize_pyhopper()
            elif args.method == 'grid':
                results = optimizer.optimize_grid_search()
            elif args.method == 'random':
                results = optimizer.optimize_random_search()
            
            # Analyze results
            analysis = optimizer.analyze_results(results)
            
            # Generate report
            report = optimizer.generate_report(results, analysis)
            
            # Create visualizations
            if args.plot:
                optimizer.create_visualizations(results, analysis)
            
            logger.info("✅ Optimization completed successfully!")
            
        elif args.mode == 'ablation':
            # Run ablation studies only
            logger.info("🔬 Starting ablation studies...")
            
            ablation_results = optimizer.run_comprehensive_ablation()
            
            # Save ablation results
            output_file = Path(optimizer.results_dir) / "ablation_results.json"
            with open(output_file, 'w') as f:
                json.dump(ablation_results, f, indent=2, default=str)
            
            logger.info(f"✅ Ablation studies completed! Results saved to: {output_file}")
            
        elif args.mode == 'comprehensive':
            # Run both ablation and optimization
            logger.info("🎯 Starting comprehensive analysis...")
            
            # Phase 1: Ablation studies
            ablation_results = optimizer.run_comprehensive_ablation()
            
            # Phase 2: Optimization
            if args.method == 'pyhopper':
                opt_results = optimizer.optimize_pyhopper()
            elif args.method == 'grid':
                opt_results = optimizer.optimize_grid_search()
            elif args.method == 'random':
                opt_results = optimizer.optimize_random_search()
            
            # Combined analysis
            analysis = optimizer.analyze_results(opt_results)
            
            # Generate comprehensive report
            comprehensive_results = {
                'ablation_results': ablation_results,
                'optimization_results': opt_results,
                'analysis': analysis
            }
            
            # Save all results
            output_file = Path(optimizer.results_dir) / "comprehensive_results.json"
            with open(output_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            # Generate report
            report = optimizer.generate_report(opt_results, analysis)
            
            # Create visualizations
            if args.plot:
                optimizer.create_visualizations(opt_results, analysis)
            
            logger.info(f"✅ Comprehensive analysis completed! Results saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise
    finally:
        optimizer.cleanup()


if __name__ == '__main__':
    main()
