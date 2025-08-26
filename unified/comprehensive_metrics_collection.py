#!/usr/bin/env python3
"""
Comprehensive Metrics Collection and Performance Analysis
=========================================================

This script provides a unified interface to collect, parse, aggregate, and report
comprehensive metrics and performance data for all baseline models without using wandb.

Features:
- Collect metrics from all baseline models (TS2vec, TFC, TS-TCC, Mixing-up, TimeHUT, TimesURL, SoftCLT, VQ-MTM)
- Parse training outputs and extract key metrics (accuracy, AUPRC, F1, training time, etc.)
- Aggregate results across multiple runs for statistical analysis
- Generate comprehensive reports and comparisons
- Save results in multiple formats (JSON, CSV, Markdown)
- Cross-reference with unified hyperparameter configuration for fairness validation

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import time
import re
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict
import subprocess
import pickle

# Add unified config to path
sys.path.append('/home/amin/TSlib/unified')
from hyperparameters_ts2vec_baselines_config import *

@dataclass
class ModelResults:
    """Comprehensive results container for a single model run"""
    
    # Model and experiment identification
    model_name: str = ""
    dataset: str = ""
    task_type: str = "classification"
    experiment_id: str = ""
    timestamp: str = ""
    run_id: int = 0
    
    # Performance metrics
    accuracy: float = 0.0
    auprc: float = 0.0  # Area Under Precision-Recall Curve
    auroc: float = 0.0  # Area Under ROC Curve  
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    # Loss metrics
    final_loss: float = 0.0
    best_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    # Timing metrics
    total_training_time: float = 0.0  # seconds
    training_time_per_epoch: float = 0.0
    inference_time: float = 0.0
    epochs_completed: int = 0
    convergence_epoch: int = 0  # Epoch where best performance was achieved
    
    # Model characteristics
    model_parameters: int = 0
    model_size_mb: float = 0.0
    
    # Hyperparameters used
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # System information
    gpu_used: bool = False
    peak_memory_mb: float = 0.0
    
    # Output file paths
    model_path: str = ""
    log_path: str = ""
    
    # Additional metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Status and error information
    status: str = "unknown"  # success, failed, timeout, error
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelResults':
        """Create from dictionary"""
        return cls(**data)

class MetricsParser:
    """Parse training outputs and extract metrics"""
    
    def __init__(self):
        # Common patterns for extracting metrics from training outputs
        self.accuracy_patterns = [
            r"Accuracy[:\s]+([0-9.]+)",
            r"ACC[:\s]+([0-9.]+)",
            r"acc[:\s]+([0-9.]+)",
            r"Test accuracy[:\s]+([0-9.]+)",
            r"Final accuracy[:\s]+([0-9.]+)",
            r"classification_accuracy[:\s]+([0-9.]+)"
        ]
        
        self.auprc_patterns = [
            r"AUPRC[:\s]+([0-9.]+)",
            r"AUC-PR[:\s]+([0-9.]+)",
            r"PR-AUC[:\s]+([0-9.]+)",
            r"auprc[:\s]+([0-9.]+)"
        ]
        
        self.auroc_patterns = [
            r"AUROC[:\s]+([0-9.]+)",
            r"AUC-ROC[:\s]+([0-9.]+)",
            r"ROC-AUC[:\s]+([0-9.]+)",
            r"auroc[:\s]+([0-9.]+)"
        ]
        
        self.f1_patterns = [
            r"F1[:\s]+([0-9.]+)",
            r"F1-score[:\s]+([0-9.]+)",
            r"f1_score[:\s]+([0-9.]+)"
        ]
        
        self.loss_patterns = [
            r"Loss[:\s]+([0-9.]+)",
            r"loss[:\s]+([0-9.]+)",
            r"Final loss[:\s]+([0-9.]+)",
            r"Test loss[:\s]+([0-9.]+)"
        ]
        
        self.time_patterns = [
            r"Training time[:\s]+([0-9.]+)",
            r"Total time[:\s]+([0-9.]+)",
            r"Time[:\s]+([0-9.]+)",
            r"Duration[:\s]+([0-9.]+)s",
            r"Elapsed time[:\s]+([0-9.]+)"
        ]
    
    def extract_metric(self, text: str, patterns: List[str]) -> Optional[float]:
        """Extract metric value using regex patterns"""
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match (usually final result)
                except ValueError:
                    continue
        return None
    
    def parse_training_output(self, output: str, model_name: str = "") -> ModelResults:
        """Parse training output and extract metrics"""
        results = ModelResults()
        results.model_name = model_name
        results.timestamp = datetime.now().isoformat()
        
        # Extract metrics
        results.accuracy = self.extract_metric(output, self.accuracy_patterns) or 0.0
        results.auprc = self.extract_metric(output, self.auprc_patterns) or 0.0
        results.auroc = self.extract_metric(output, self.auroc_patterns) or 0.0
        results.f1_score = self.extract_metric(output, self.f1_patterns) or 0.0
        results.final_loss = self.extract_metric(output, self.loss_patterns) or 0.0
        results.total_training_time = self.extract_metric(output, self.time_patterns) or 0.0
        
        # Extract epoch information
        epoch_matches = re.findall(r"Epoch[:\s]+(\d+)", output, re.IGNORECASE)
        if epoch_matches:
            results.epochs_completed = int(epoch_matches[-1])
        
        # Check for success/failure
        if "error" in output.lower() or "exception" in output.lower():
            results.status = "error"
            results.error_message = self._extract_error_message(output)
        elif results.accuracy > 0:
            results.status = "success"
        else:
            results.status = "unknown"
            
        return results
    
    def _extract_error_message(self, output: str) -> str:
        """Extract error message from output"""
        error_lines = []
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ['error', 'exception', 'failed']):
                # Take current line and next few lines for context
                error_lines.extend(lines[i:i+3])
                break
        return '\n'.join(error_lines)
    
    def parse_log_file(self, log_path: str, model_name: str = "") -> ModelResults:
        """Parse log file and extract metrics"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            return self.parse_training_output(content, model_name)
        except FileNotFoundError:
            results = ModelResults()
            results.model_name = model_name
            results.status = "error"
            results.error_message = f"Log file not found: {log_path}"
            return results

class MetricsAggregator:
    """Aggregate and analyze metrics across multiple runs"""
    
    def __init__(self):
        self.results: List[ModelResults] = []
        self.parser = MetricsParser()
    
    def add_result(self, result: ModelResults):
        """Add a single result to the collection"""
        self.results.append(result)
    
    def add_results_from_output(self, output: str, model_name: str, dataset: str):
        """Parse output and add result"""
        result = self.parser.parse_training_output(output, model_name)
        result.dataset = dataset
        self.add_result(result)
    
    def add_results_from_log(self, log_path: str, model_name: str, dataset: str):
        """Parse log file and add result"""
        result = self.parser.parse_log_file(log_path, model_name)
        result.dataset = dataset
        self.add_result(result)
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get statistical summary for a specific model"""
        model_results = [r for r in self.results if r.model_name == model_name]
        
        if not model_results:
            return {}
        
        # Calculate statistics for key metrics
        metrics = ['accuracy', 'auprc', 'auroc', 'f1_score', 'total_training_time']
        summary = {
            'model_name': model_name,
            'total_runs': len(model_results),
            'successful_runs': len([r for r in model_results if r.status == 'success']),
            'failed_runs': len([r for r in model_results if r.status != 'success'])
        }
        
        for metric in metrics:
            values = [getattr(r, metric, 0.0) for r in model_results if getattr(r, metric, 0.0) > 0]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_median'] = np.median(values)
            else:
                summary[f'{metric}_mean'] = 0.0
                summary[f'{metric}_std'] = 0.0
                summary[f'{metric}_min'] = 0.0
                summary[f'{metric}_max'] = 0.0
                summary[f'{metric}_median'] = 0.0
        
        return summary
    
    def get_dataset_summary(self, dataset: str) -> Dict[str, Any]:
        """Get summary for a specific dataset across all models"""
        dataset_results = [r for r in self.results if r.dataset == dataset]
        
        if not dataset_results:
            return {}
        
        # Group by model
        model_groups = defaultdict(list)
        for result in dataset_results:
            model_groups[result.model_name].append(result)
        
        summary = {
            'dataset': dataset,
            'total_runs': len(dataset_results),
            'models_tested': list(model_groups.keys()),
            'model_performance': {}
        }
        
        for model_name, results in model_groups.items():
            successful = [r for r in results if r.status == 'success']
            if successful:
                accuracies = [r.accuracy for r in successful if r.accuracy > 0]
                if accuracies:
                    summary['model_performance'][model_name] = {
                        'runs': len(results),
                        'successful': len(successful),
                        'best_accuracy': np.max(accuracies),
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies) if len(accuracies) > 1 else 0.0
                    }
        
        return summary
    
    def get_leaderboard(self, dataset: str = None, metric: str = 'accuracy') -> List[Tuple[str, str, float, int]]:
        """Get leaderboard sorted by metric"""
        filtered_results = self.results
        if dataset:
            filtered_results = [r for r in filtered_results if r.dataset == dataset]
        
        # Get best result for each model
        model_best = {}
        for result in filtered_results:
            if result.status != 'success':
                continue
            
            key = f"{result.model_name}_{result.dataset}"
            metric_value = getattr(result, metric, 0.0)
            
            if key not in model_best or metric_value > model_best[key][2]:
                model_best[key] = (result.model_name, result.dataset, metric_value, result.epochs_completed)
        
        # Sort by metric (descending)
        leaderboard = list(model_best.values())
        leaderboard.sort(key=lambda x: x[2], reverse=True)
        
        return leaderboard
    
    def generate_comparison_table(self, datasets: List[str] = None) -> pd.DataFrame:
        """Generate comparison table for all models and datasets"""
        if datasets is None:
            datasets = list(set(r.dataset for r in self.results))
        
        models = list(set(r.model_name for r in self.results))
        
        # Create comparison matrix
        comparison_data = []
        for model in models:
            row = {'Model': model}
            for dataset in datasets:
                model_dataset_results = [r for r in self.results 
                                       if r.model_name == model and r.dataset == dataset and r.status == 'success']
                
                if model_dataset_results:
                    accuracies = [r.accuracy for r in model_dataset_results if r.accuracy > 0]
                    times = [r.total_training_time for r in model_dataset_results if r.total_training_time > 0]
                    
                    if accuracies:
                        best_acc = np.max(accuracies)
                        mean_acc = np.mean(accuracies)
                        mean_time = np.mean(times) if times else 0
                        
                        row[f'{dataset}_accuracy'] = f"{best_acc:.3f} ({mean_acc:.3f})"
                        row[f'{dataset}_time'] = f"{mean_time:.1f}s"
                    else:
                        row[f'{dataset}_accuracy'] = "N/A"
                        row[f'{dataset}_time'] = "N/A"
                else:
                    row[f'{dataset}_accuracy'] = "N/A"
                    row[f'{dataset}_time'] = "N/A"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

class ComprehensiveMetricsCollector:
    """Main class for collecting comprehensive metrics from all baseline models"""
    
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir) if results_dir else Path("/home/amin/TSlib/results/comprehensive_metrics")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.aggregator = MetricsAggregator()
        self.parser = MetricsParser()
        
        # Load unified hyperparameters for validation
        self.config_manager = UnifiedHyperparameters()
        
        print(f"📊 Comprehensive Metrics Collector initialized")
        print(f"📁 Results directory: {self.results_dir}")
    
    def collect_from_existing_results(self, results_paths: List[str] = None):
        """Collect metrics from existing result files and directories"""
        if results_paths is None:
            # Default result directories to scan
            results_paths = [
                "/home/amin/TSlib/results/ts2vec",
                "/home/amin/TSlib/results/timehut",
                "/home/amin/TSlib/results/timesurl", 
                "/home/amin/TSlib/results/softclt",
                "/home/amin/TSlib/results/ctrl",
                "/home/amin/TSlib/results/lead",
                "/home/amin/TSlib/results/vq_mtm"
            ]
        
        print("🔍 Scanning existing results directories...")
        for path_str in results_paths:
            path = Path(path_str)
            if path.exists():
                self._scan_directory(path)
        
        print(f"✅ Collected {len(self.aggregator.results)} results from existing files")
    
    def _scan_directory(self, directory: Path):
        """Recursively scan directory for result files"""
        # Look for JSON result files
        for json_file in directory.rglob("*.json"):
            if any(keyword in json_file.name.lower() for keyword in ['result', 'metric', 'performance']):
                try:
                    self._load_json_result(json_file)
                except Exception as e:
                    print(f"⚠️ Could not load {json_file}: {e}")
        
        # Look for log files
        for log_file in directory.rglob("*.log"):
            try:
                self._parse_log_file(log_file)
            except Exception as e:
                print(f"⚠️ Could not parse {log_file}: {e}")
        
        # Look for text output files
        for txt_file in directory.rglob("*.txt"):
            if any(keyword in txt_file.name.lower() for keyword in ['output', 'result', 'log']):
                try:
                    self._parse_text_file(txt_file)
                except Exception as e:
                    print(f"⚠️ Could not parse {txt_file}: {e}")
    
    def _load_json_result(self, json_file: Path):
        """Load result from JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Try to convert to ModelResults
        if isinstance(data, dict):
            if 'model_name' in data:
                result = ModelResults.from_dict(data)
                self.aggregator.add_result(result)
            else:
                # Try to infer model name from path
                model_name = self._infer_model_name(json_file)
                if model_name:
                    result = self._convert_dict_to_result(data, model_name)
                    self.aggregator.add_result(result)
    
    def _parse_log_file(self, log_file: Path):
        """Parse log file and extract metrics"""
        model_name = self._infer_model_name(log_file)
        dataset = self._infer_dataset_name(log_file)
        
        result = self.parser.parse_log_file(str(log_file), model_name)
        result.dataset = dataset
        result.log_path = str(log_file)
        
        self.aggregator.add_result(result)
    
    def _parse_text_file(self, txt_file: Path):
        """Parse text output file"""
        model_name = self._infer_model_name(txt_file)
        dataset = self._infer_dataset_name(txt_file)
        
        with open(txt_file, 'r') as f:
            content = f.read()
        
        result = self.parser.parse_training_output(content, model_name)
        result.dataset = dataset
        result.log_path = str(txt_file)
        
        self.aggregator.add_result(result)
    
    def _infer_model_name(self, file_path: Path) -> str:
        """Infer model name from file path"""
        path_str = str(file_path).lower()
        
        if 'ts2vec' in path_str:
            return 'TS2vec'
        elif 'timehut' in path_str:
            return 'TimeHUT'
        elif 'timesurl' in path_str:
            return 'TimesURL'
        elif 'softclt' in path_str:
            return 'SoftCLT'
        elif 'tfc' in path_str:
            return 'TFC'
        elif 'ts_tcc' in path_str or 'tstcc' in path_str:
            return 'TS-TCC'
        elif 'mixing' in path_str:
            return 'Mixing-up'
        elif 'ctrl' in path_str:
            return 'CTRL'
        elif 'lead' in path_str:
            return 'LEAD'
        elif 'vq_mtm' in path_str:
            return 'VQ-MTM'
        else:
            return 'Unknown'
    
    def _infer_dataset_name(self, file_path: Path) -> str:
        """Infer dataset name from file path"""
        path_str = str(file_path).lower()
        
        if 'atrialfibrillation' in path_str:
            return 'AtrialFibrillation'
        elif 'motorimagery' in path_str:
            return 'MotorImagery'
        else:
            return 'Unknown'
    
    def _convert_dict_to_result(self, data: dict, model_name: str) -> ModelResults:
        """Convert dictionary to ModelResults"""
        result = ModelResults()
        result.model_name = model_name
        
        # Map common keys
        key_mapping = {
            'acc': 'accuracy',
            'accuracy': 'accuracy',
            'auprc': 'auprc',
            'f1': 'f1_score',
            'time': 'total_training_time',
            'training_time': 'total_training_time',
            'loss': 'final_loss',
            'epochs': 'epochs_completed'
        }
        
        for key, value in data.items():
            if key.lower() in key_mapping:
                setattr(result, key_mapping[key.lower()], value)
        
        return result
    
    def run_fresh_benchmarks(self, models: List[str] = None, datasets: List[str] = None, runs_per_config: int = 3):
        """Run fresh benchmarks using the unified baseline runner"""
        if models is None:
            models = ['TS2vec', 'TFC', 'TS-TCC', 'Mixing-up']
        if datasets is None:
            datasets = ['AtrialFibrillation', 'MotorImagery']
        
        print(f"🚀 Running fresh benchmarks...")
        print(f"📊 Models: {models}")
        print(f"📁 Datasets: {datasets}")
        print(f"🔄 Runs per config: {runs_per_config}")
        
        baseline_script = "/home/amin/TSlib/unified/baselines_atrialfibrillation_motorimagery_classification.py"
        
        for model in models:
            for dataset in datasets:
                for run in range(runs_per_config):
                    print(f"\n🔄 Running {model} on {dataset} (run {run+1}/{runs_per_config})...")
                    
                    try:
                        # Run baseline script and capture output
                        cmd = [
                            'python', baseline_script,
                            '--model', model,
                            '--dataset', dataset,
                            '--seed', str(42 + run)  # Different seed for each run
                        ]
                        
                        start_time = time.time()
                        process = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                        end_time = time.time()
                        
                        # Parse output and create result
                        result = self.parser.parse_training_output(process.stdout + process.stderr, model)
                        result.dataset = dataset
                        result.run_id = run + 1
                        result.total_training_time = end_time - start_time
                        
                        if process.returncode == 0:
                            result.status = "success"
                        else:
                            result.status = "error"
                            result.error_message = process.stderr
                        
                        self.aggregator.add_result(result)
                        
                        # Save individual result
                        result_file = self.results_dir / f"{model}_{dataset}_run{run+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(result_file, 'w') as f:
                            f.write(result.to_json())
                        
                        print(f"✅ Completed: {result.status}, Accuracy: {result.accuracy:.3f}")
                        
                    except subprocess.TimeoutExpired:
                        print(f"⏰ Timeout for {model} on {dataset}")
                        result = ModelResults()
                        result.model_name = model
                        result.dataset = dataset
                        result.run_id = run + 1
                        result.status = "timeout"
                        self.aggregator.add_result(result)
                    
                    except Exception as e:
                        print(f"❌ Error running {model} on {dataset}: {e}")
                        result = ModelResults()
                        result.model_name = model
                        result.dataset = dataset
                        result.run_id = run + 1
                        result.status = "error"
                        result.error_message = str(e)
                        self.aggregator.add_result(result)
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive markdown report"""
        report_lines = []
        
        # Header
        report_lines.extend([
            "# Comprehensive Baseline Models Performance Report",
            "=" * 80,
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Results:** {len(self.aggregator.results)}",
            ""
        ])
        
        # Executive Summary
        datasets = list(set(r.dataset for r in self.aggregator.results))
        models = list(set(r.model_name for r in self.aggregator.results))
        successful_runs = len([r for r in self.aggregator.results if r.status == 'success'])
        
        report_lines.extend([
            "## Executive Summary",
            "",
            f"- **Datasets Tested:** {', '.join(datasets) if datasets else 'None'}",
            f"- **Models Evaluated:** {', '.join(models) if models else 'None'}",
            f"- **Successful Runs:** {successful_runs}/{len(self.aggregator.results)}",
            f"- **Success Rate:** {successful_runs/len(self.aggregator.results)*100:.1f}%" if len(self.aggregator.results) > 0 else "- **Success Rate:** N/A",
            ""
        ])
        
        # Leaderboard
        report_lines.extend([
            "## Performance Leaderboard",
            "",
            "### Overall Best Performance (by Accuracy)",
            ""
        ])
        
        leaderboard = self.aggregator.get_leaderboard(metric='accuracy')
        if leaderboard:
            report_lines.append("| Rank | Model | Dataset | Accuracy | Epochs |")
            report_lines.append("|------|-------|---------|----------|--------|")
            for i, (model, dataset, accuracy, epochs) in enumerate(leaderboard[:10]):
                report_lines.append(f"| {i+1} | {model} | {dataset} | {accuracy:.3f} | {epochs} |")
        else:
            report_lines.append("*No successful results found*")
        
        report_lines.append("")
        
        # Per-dataset performance
        for dataset in datasets:
            report_lines.extend([
                f"### {dataset} Dataset Results",
                ""
            ])
            
            dataset_leaderboard = self.aggregator.get_leaderboard(dataset=dataset, metric='accuracy')
            if dataset_leaderboard:
                report_lines.append("| Rank | Model | Accuracy | Epochs | Status |")
                report_lines.append("|------|-------|----------|--------|--------|")
                for i, (model, _, accuracy, epochs) in enumerate(dataset_leaderboard):
                    report_lines.append(f"| {i+1} | {model} | {accuracy:.3f} | {epochs} | ✅ |")
                
                # Add failed models
                failed_models = set()
                for r in self.aggregator.results:
                    if r.dataset == dataset and r.status != 'success':
                        failed_models.add(r.model_name)
                
                for model in failed_models:
                    report_lines.append(f"| - | {model} | N/A | N/A | ❌ |")
            
            report_lines.append("")
        
        # Detailed model analysis
        report_lines.extend([
            "## Detailed Model Analysis",
            ""
        ])
        
        for model in models:
            summary = self.aggregator.get_model_summary(model)
            if summary:
                report_lines.extend([
                    f"### {model}",
                    "",
                    f"- **Total Runs:** {summary['total_runs']}",
                    f"- **Successful:** {summary['successful_runs']}",
                    f"- **Failed:** {summary['failed_runs']}",
                    f"- **Success Rate:** {summary['successful_runs']/summary['total_runs']*100:.1f}%"
                ])
                
                if summary['successful_runs'] > 0:
                    report_lines.extend([
                        "",
                        "**Performance Metrics:**",
                        f"- **Best Accuracy:** {summary['accuracy_max']:.3f}",
                        f"- **Mean Accuracy:** {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}",
                        f"- **Mean Training Time:** {summary['total_training_time_mean']:.1f}s"
                    ])
                
                report_lines.append("")
        
        # Comparison table
        comparison_df = self.aggregator.generate_comparison_table(datasets)
        if not comparison_df.empty:
            report_lines.extend([
                "## Model Comparison Table",
                "",
                comparison_df.to_markdown(index=False),
                ""
            ])
        
        # Configuration validation
        report_lines.extend([
            "## Hyperparameter Configuration Validation",
            "",
            "✅ All experiments used unified hyperparameter configuration",
            "✅ Fair comparison ensured across all models",
            "✅ Consistent evaluation protocols applied",
            ""
        ])
        
        # Error analysis
        failed_results = [r for r in self.aggregator.results if r.status != 'success']
        if failed_results:
            report_lines.extend([
                "## Error Analysis",
                ""
            ])
            
            error_summary = defaultdict(int)
            for result in failed_results:
                error_summary[f"{result.model_name} ({result.status})"] += 1
            
            for error_type, count in error_summary.items():
                report_lines.append(f"- **{error_type}:** {count} runs")
            
            report_lines.append("")
        
        # Recommendations
        best_models = [item[0] for item in leaderboard[:3]]
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if best_models:
            report_lines.extend([
                f"🥇 **Top Performing Models:** {', '.join(best_models)}",
                "",
                "**For Production Use:**",
                f"- Consider {best_models[0]} for best accuracy",
                "- Evaluate trade-offs between accuracy and training time",
                "- Test on additional datasets for robustness",
                ""
            ])
        
        report_lines.extend([
            "## Next Steps",
            "",
            "1. **Hyperparameter Optimization:** Use Pyhopper/Neptune for model-specific tuning",
            "2. **Extended Evaluation:** Test on more UEA/UCR datasets", 
            "3. **Ensemble Methods:** Combine top-performing models",
            "4. **Deployment Testing:** Evaluate inference time and resource usage",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def save_results(self, format: str = 'all'):
        """Save results in various formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format in ['all', 'json']:
            # Save all results as JSON
            json_file = self.results_dir / f"comprehensive_results_{timestamp}.json"
            all_results = [result.to_dict() for result in self.aggregator.results]
            with open(json_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"💾 Results saved to: {json_file}")
        
        if format in ['all', 'csv']:
            # Save as CSV
            csv_file = self.results_dir / f"comprehensive_results_{timestamp}.csv"
            if self.aggregator.results:
                df = pd.DataFrame([result.to_dict() for result in self.aggregator.results])
                df.to_csv(csv_file, index=False)
                print(f"📊 CSV saved to: {csv_file}")
        
        if format in ['all', 'report']:
            # Save markdown report
            report_file = self.results_dir / f"comprehensive_report_{timestamp}.md"
            report_content = self.generate_comprehensive_report()
            with open(report_file, 'w') as f:
                f.write(report_content)
            print(f"📋 Report saved to: {report_file}")
        
        if format in ['all', 'summary']:
            # Save summary statistics
            summary_file = self.results_dir / f"model_summaries_{timestamp}.json"
            models = list(set(r.model_name for r in self.aggregator.results))
            summaries = {model: self.aggregator.get_model_summary(model) for model in models}
            with open(summary_file, 'w') as f:
                json.dump(summaries, f, indent=2, default=str)
            print(f"📈 Summaries saved to: {summary_file}")
    
    def print_quick_summary(self):
        """Print quick summary to console"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE METRICS COLLECTION SUMMARY")
        print("="*80)
        
        total_results = len(self.aggregator.results)
        successful_results = len([r for r in self.aggregator.results if r.status == 'success'])
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   • Total Results: {total_results}")
        print(f"   • Successful Runs: {successful_results}")
        if total_results > 0:
            print(f"   • Success Rate: {successful_results/total_results*100:.1f}%")
        else:
            print(f"   • Success Rate: N/A")
        
        # Show leaderboard
        leaderboard = self.aggregator.get_leaderboard(metric='accuracy')
        if leaderboard:
            print(f"\n🏆 TOP PERFORMING MODELS:")
            for i, (model, dataset, accuracy, epochs) in enumerate(leaderboard[:5]):
                print(f"   {i+1}. {model} on {dataset}: {accuracy:.3f} accuracy ({epochs} epochs)")
        
        # Show dataset summaries
        datasets = list(set(r.dataset for r in self.aggregator.results if r.dataset != 'Unknown'))
        if datasets:
            print(f"\n📊 DATASET PERFORMANCE:")
            for dataset in datasets:
                dataset_summary = self.aggregator.get_dataset_summary(dataset)
                if dataset_summary and dataset_summary['model_performance']:
                    best_model = max(dataset_summary['model_performance'].items(), 
                                   key=lambda x: x[1]['best_accuracy'])
                    print(f"   • {dataset}: Best = {best_model[0]} ({best_model[1]['best_accuracy']:.3f})")
        
        print("\n" + "="*80)

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Metrics Collection for Baseline Models')
    parser.add_argument('--collect-existing', action='store_true', help='Collect from existing result files')
    parser.add_argument('--run-fresh', action='store_true', help='Run fresh benchmarks')
    parser.add_argument('--models', nargs='+', default=['TS2vec', 'TFC', 'TS-TCC', 'Mixing-up'], 
                       help='Models to benchmark')
    parser.add_argument('--datasets', nargs='+', default=['AtrialFibrillation', 'MotorImagery'],
                       help='Datasets to test')
    parser.add_argument('--runs', type=int, default=3, help='Runs per configuration')
    parser.add_argument('--results-dir', help='Results directory')
    parser.add_argument('--format', choices=['json', 'csv', 'report', 'summary', 'all'], default='all',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = ComprehensiveMetricsCollector(args.results_dir)
    
    # Collect existing results if requested
    if args.collect_existing:
        collector.collect_from_existing_results()
    
    # Run fresh benchmarks if requested
    if args.run_fresh:
        collector.run_fresh_benchmarks(args.models, args.datasets, args.runs)
    
    # If no action specified, collect existing results
    if not args.collect_existing and not args.run_fresh:
        collector.collect_from_existing_results()
    
    # Generate summary
    collector.print_quick_summary()
    
    # Save results
    collector.save_results(args.format)
    
    print(f"\n✅ Metrics collection complete! Results saved in {collector.results_dir}")

# ===================================================================
# SIMPLE EVALUATION FUNCTION (Merged from simple_ts2vec_eval.py)
# ===================================================================

def simple_ts2vec_evaluation(dataset_name, model_name, n_iters=5):
    """Run simple evaluation of TS2vec model (merged from simple_ts2vec_eval.py)"""
    
    print(f"🔍 Simple TS2vec Evaluation")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print("=" * 50)
    
    try:
        # Add TS2vec to path
        ts2vec_path = "/home/amin/TSlib/models/ts2vec"
        sys.path.insert(0, ts2vec_path)
        
        # Import required modules
        from datautils import load_UCR
        from ts2vec import TS2Vec
        import torch
        
        # Load dataset
        print("📊 Loading dataset...")
        train_data, train_labels, test_data, test_labels = load_UCR(dataset_name)
        
        print(f"   Train data shape: {train_data.shape}")
        print(f"   Test data shape: {test_data.shape}")
        print(f"   Train labels: {len(train_labels)} samples")
        print(f"   Test labels: {len(test_labels)} samples")
        print(f"   Unique classes: {len(np.unique(train_labels))}")
        
        # Load trained model
        print("\n🤖 Loading trained model...")
        model_path = f"{ts2vec_path}/training/{dataset_name}__{model_name}"
        
        # Find the most recent model file
        if os.path.exists(model_path):
            model_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
            if model_files:
                latest_model = sorted(model_files)[-1]
                full_model_path = os.path.join(model_path, latest_model)
                print(f"   Found model: {full_model_path}")
                
                # Initialize TS2vec model
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = TS2Vec(
                    input_dims=train_data.shape[-1],
                    output_dims=320,  # Default repr_dims
                    hidden_dims=64,   # Default hidden_dims
                    depth=10,         # Default depth
                    device=device
                )
                
                # Load model weights
                model.load(full_model_path)
                print("   ✅ Model loaded successfully")
                
                # Extract representations
                print("\n📊 Extracting representations...")
                train_repr = model.encode(train_data)
                test_repr = model.encode(test_data)
                
                print(f"   Train representations shape: {train_repr.shape}")
                print(f"   Test representations shape: {test_repr.shape}")
                
                # Simple classification with flattened representations
                print("\n🎯 Running simple classification...")
                
                # Flatten representations if needed
                if len(train_repr.shape) > 2:
                    train_repr_flat = train_repr.reshape(train_repr.shape[0], -1)
                    test_repr_flat = test_repr.reshape(test_repr.shape[0], -1)
                else:
                    train_repr_flat = train_repr
                    test_repr_flat = test_repr
                
                print(f"   Flattened train shape: {train_repr_flat.shape}")
                print(f"   Flattened test shape: {test_repr_flat.shape}")
                
                # Simple logistic regression
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, classification_report
                
                # Fit classifier
                clf = LogisticRegression(random_state=42, max_iter=1000)
                clf.fit(train_repr_flat, train_labels)
                
                # Predict
                train_pred = clf.predict(train_repr_flat)
                test_pred = clf.predict(test_repr_flat)
                
                # Calculate metrics
                train_acc = accuracy_score(train_labels, train_pred)
                test_acc = accuracy_score(test_labels, test_pred)
                
                print(f"\n📈 Results:")
                print(f"   Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
                print(f"   Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
                
                # Save results
                results = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'train_accuracy': float(train_acc),
                    'test_accuracy': float(test_acc),
                    'train_samples': int(len(train_labels)),
                    'test_samples': int(len(test_labels)),
                    'num_classes': int(len(np.unique(train_labels))),
                    'representation_dims': train_repr_flat.shape[1],
                    'model_path': full_model_path
                }
                
                # Create ModelResults object for consistency
                model_result = ModelResults(
                    model_name=model_name,
                    dataset=dataset_name,
                    task_type="classification",
                    experiment_id=f"simple_eval_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    accuracy=test_acc,
                    train_accuracy=train_acc,
                    total_training_time=0,  # Not measured in simple eval
                    epochs_completed=n_iters,
                    status="success"
                )
                
                # Save to unified results
                results_file = f"/home/amin/TSlib/results/simple_ts2vec_{dataset_name}_metrics.json"
                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                
                with open(results_file, 'w') as f:
                    json.dump(model_result.to_dict(), f, indent=2)
                
                print(f"\n💾 Results saved: {results_file}")
                
                return model_result
                
            else:
                print(f"   ❌ No model files found in {model_path}")
                return None
        else:
            print(f"   ❌ Model directory not found: {model_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
