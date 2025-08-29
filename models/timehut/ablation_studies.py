#!/usr/bin/env python3
"""
TimeHUT Systematic Ablation Studies Framework
===========================================

Comprehensive ablation study framework for TimeHUT model components.
Provides systematic testing of individual components and their interactions.

Author: TimeHUT Analysis Framework
Date: August 26, 2025
"""

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class TimeHUTAblationFramework:
    """Systematic ablation study framework for TimeHUT"""
    
    def __init__(self, base_dir: str = None, results_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path("/home/amin/TSlib")
        self.timehut_dir = self.base_dir / "models" / "timehut"
        self.results_dir = Path(results_dir) if results_dir else self.timehut_dir / "analysis_results"
        self.ablation_results_dir = self.results_dir / "ablations"
        self.ablation_results_dir.mkdir(exist_ok=True)
        
        self.ablation_results = {}
        self.statistical_results = {}
        
    def design_component_ablations(self) -> Dict[str, Any]:
        """Design comprehensive component ablation experiments"""
        
        ablation_design = {
            "component_ablations": {
                "baseline_comparison": {
                    "description": "Compare TimeHUT components against baseline TS2Vec",
                    "scenarios": [
                        {
                            "name": "pure_ts2vec",
                            "description": "Pure TS2Vec without any TimeHUT enhancements",
                            "config": {
                                "amc_instance": 0.0,
                                "amc_temporal": 0.0, 
                                "temperature_scheduling": False,
                                "enhanced_losses": False
                            }
                        },
                        {
                            "name": "timehut_minimal",
                            "description": "TimeHUT with minimal enhancements",
                            "config": {
                                "amc_instance": 0.1,
                                "amc_temporal": 0.1,
                                "temperature_scheduling": False,
                                "enhanced_losses": True
                            }
                        },
                        {
                            "name": "timehut_full",
                            "description": "TimeHUT with all enhancements",
                            "config": {
                                "amc_instance": 1.0,
                                "amc_temporal": 0.5,
                                "temperature_scheduling": True,
                                "enhanced_losses": True,
                                "min_tau": 0.15,
                                "max_tau": 0.75,
                                "t_max": 10.5
                            }
                        }
                    ]
                },
                
                "amc_component_analysis": {
                    "description": "Individual AMC component effects",
                    "components": [
                        {
                            "name": "instance_only",
                            "description": "Only instance-wise contrastive loss",
                            "config": {
                                "amc_instance": 1.0,
                                "amc_temporal": 0.0,
                                "amc_margin": 0.5
                            }
                        },
                        {
                            "name": "temporal_only", 
                            "description": "Only temporal contrastive loss",
                            "config": {
                                "amc_instance": 0.0,
                                "amc_temporal": 1.0,
                                "amc_margin": 0.5
                            }
                        },
                        {
                            "name": "both_amc",
                            "description": "Both instance and temporal AMC losses",
                            "config": {
                                "amc_instance": 1.0,
                                "amc_temporal": 1.0,
                                "amc_margin": 0.5
                            }
                        },
                        {
                            "name": "margin_sensitivity",
                            "description": "Test different margin values",
                            "parameter_sweep": {
                                "amc_margin": [0.1, 0.3, 0.5, 0.7, 1.0],
                                "amc_instance": [1.0],
                                "amc_temporal": [0.5]
                            }
                        }
                    ]
                },
                
                "temperature_scheduler_analysis": {
                    "description": "Temperature scheduling method comparison",
                    "schedulers": [
                        {
                            "name": "no_scheduling",
                            "description": "No temperature scheduling (baseline)",
                            "config": {"temperature_scheduling": False}
                        },
                        {
                            "name": "linear_scheduling",
                            "description": "Linear temperature decay",
                            "config": {
                                "temperature_scheduling": True,
                                "scheduler_method": "linear",
                                "min_tau": 0.1,
                                "max_tau": 0.8
                            }
                        },
                        {
                            "name": "cosine_scheduling",
                            "description": "Cosine annealing temperature",
                            "config": {
                                "temperature_scheduling": True,
                                "scheduler_method": "cosine",
                                "min_tau": 0.1,
                                "max_tau": 0.8,
                                "t_max": 10.0
                            }
                        },
                        {
                            "name": "exponential_scheduling",
                            "description": "Exponential temperature decay",
                            "config": {
                                "temperature_scheduling": True,
                                "scheduler_method": "exponential",
                                "min_tau": 0.1,
                                "max_tau": 0.8,
                                "decay_rate": 0.95
                            }
                        },
                        {
                            "name": "sigmoid_scheduling",
                            "description": "Sigmoid temperature scheduling",
                            "config": {
                                "temperature_scheduling": True,
                                "scheduler_method": "sigmoid",
                                "min_tau": 0.1,
                                "max_tau": 0.8,
                                "steepness": 5.0
                            }
                        }
                    ]
                },
                
                "loss_component_ablations": {
                    "description": "Individual loss component effects",
                    "loss_components": [
                        {
                            "name": "instance_contrastive_only",
                            "description": "Only instance contrastive loss",
                            "config": {
                                "enable_instance_contrastive": True,
                                "enable_temporal_contrastive": False,
                                "enable_instance_discriminative": False,
                                "enable_temporal_discriminative": False
                            }
                        },
                        {
                            "name": "temporal_contrastive_only",
                            "description": "Only temporal contrastive loss",
                            "config": {
                                "enable_instance_contrastive": False,
                                "enable_temporal_contrastive": True,
                                "enable_instance_discriminative": False,
                                "enable_temporal_discriminative": False
                            }
                        },
                        {
                            "name": "all_losses",
                            "description": "All loss components enabled",
                            "config": {
                                "enable_instance_contrastive": True,
                                "enable_temporal_contrastive": True,
                                "enable_instance_discriminative": True,
                                "enable_temporal_discriminative": True
                            }
                        }
                    ]
                }
            },
            
            "interaction_ablations": {
                "description": "Component interaction effects",
                "interactions": [
                    {
                        "name": "amc_temp_interaction",
                        "description": "AMC and temperature scheduling interaction",
                        "factor_grid": {
                            "amc_levels": [0.0, 0.5, 1.0, 2.0],
                            "temp_scheduling": [False, True],
                            "temp_methods": ["linear", "cosine"]
                        }
                    },
                    {
                        "name": "loss_weight_interactions",
                        "description": "Loss component weight interactions",
                        "weight_combinations": [
                            {"instance": 1.0, "temporal": 0.0},
                            {"instance": 0.0, "temporal": 1.0},
                            {"instance": 1.0, "temporal": 1.0},
                            {"instance": 2.0, "temporal": 1.0},
                            {"instance": 1.0, "temporal": 2.0}
                        ]
                    }
                ]
            },
            
            "sensitivity_analysis": {
                "description": "Parameter sensitivity analysis",
                "parameters": {
                    "amc_instance": {
                        "range": [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                        "baseline": 1.0,
                        "description": "Instance AMC coefficient sensitivity"
                    },
                    "amc_temporal": {
                        "range": [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                        "baseline": 0.5,
                        "description": "Temporal AMC coefficient sensitivity"
                    },
                    "amc_margin": {
                        "range": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                        "baseline": 0.5,
                        "description": "AMC margin sensitivity"
                    },
                    "min_tau": {
                        "range": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                        "baseline": 0.15,
                        "description": "Minimum temperature sensitivity"
                    },
                    "max_tau": {
                        "range": [0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
                        "baseline": 0.75,
                        "description": "Maximum temperature sensitivity"
                    }
                }
            }
        }
        
        return ablation_design
    
    def generate_ablation_experiments(self, ablation_design: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate concrete experiment configurations from ablation design"""
        
        experiments = []
        
        # Component ablations
        component_ablations = ablation_design["component_ablations"]
        
        # Baseline comparison experiments
        for scenario in component_ablations["baseline_comparison"]["scenarios"]:
            experiments.append({
                "experiment_type": "baseline_comparison",
                "scenario_name": scenario["name"],
                "description": scenario["description"],
                "config": scenario["config"],
                "datasets": ["Chinatown", "AtrialFibrillation", "SyntheticControl"],
                "epochs": 200,
                "repeats": 5
            })
        
        # AMC component experiments
        for component in component_ablations["amc_component_analysis"]["components"]:
            if "parameter_sweep" in component:
                # Generate parameter sweep experiments
                param_sweep = component["parameter_sweep"]
                param_names = list(param_sweep.keys())
                param_combinations = list(itertools.product(*param_sweep.values()))
                
                for param_combo in param_combinations:
                    config = dict(zip(param_names, param_combo))
                    experiments.append({
                        "experiment_type": "amc_component",
                        "scenario_name": f"{component['name']}_{param_combo}",
                        "description": component["description"],
                        "config": config,
                        "datasets": ["Chinatown", "SyntheticControl"],
                        "epochs": 150,
                        "repeats": 3
                    })
            else:
                experiments.append({
                    "experiment_type": "amc_component",
                    "scenario_name": component["name"],
                    "description": component["description"],
                    "config": component["config"],
                    "datasets": ["Chinatown", "AtrialFibrillation"],
                    "epochs": 150,
                    "repeats": 3
                })
        
        # Temperature scheduler experiments
        for scheduler in component_ablations["temperature_scheduler_analysis"]["schedulers"]:
            experiments.append({
                "experiment_type": "temperature_scheduler",
                "scenario_name": scheduler["name"],
                "description": scheduler["description"],
                "config": scheduler["config"],
                "datasets": ["Chinatown", "SyntheticControl"],
                "epochs": 200,
                "repeats": 3
            })
        
        # Loss component experiments
        for loss_comp in component_ablations["loss_component_ablations"]["loss_components"]:
            experiments.append({
                "experiment_type": "loss_component",
                "scenario_name": loss_comp["name"],
                "description": loss_comp["description"],
                "config": loss_comp["config"],
                "datasets": ["Chinatown"],
                "epochs": 100,
                "repeats": 3
            })
        
        # Interaction experiments
        interaction_ablations = ablation_design["interaction_ablations"]
        
        # AMC-Temperature interactions
        amc_temp_interaction = interaction_ablations["interactions"][0]
        factor_grid = amc_temp_interaction["factor_grid"]
        
        for amc_level in factor_grid["amc_levels"]:
            for temp_enabled in factor_grid["temp_scheduling"]:
                if temp_enabled:
                    for temp_method in factor_grid["temp_methods"]:
                        experiments.append({
                            "experiment_type": "amc_temp_interaction",
                            "scenario_name": f"amc{amc_level}_temp_{temp_method}",
                            "description": "AMC-temperature interaction",
                            "config": {
                                "amc_instance": amc_level,
                                "amc_temporal": amc_level * 0.5,
                                "temperature_scheduling": True,
                                "scheduler_method": temp_method,
                                "min_tau": 0.15,
                                "max_tau": 0.75
                            },
                            "datasets": ["Chinatown"],
                            "epochs": 150,
                            "repeats": 3
                        })
                else:
                    experiments.append({
                        "experiment_type": "amc_temp_interaction",
                        "scenario_name": f"amc{amc_level}_no_temp",
                        "description": "AMC without temperature",
                        "config": {
                            "amc_instance": amc_level,
                            "amc_temporal": amc_level * 0.5,
                            "temperature_scheduling": False
                        },
                        "datasets": ["Chinatown"],
                        "epochs": 150,
                        "repeats": 3
                    })
        
        # Sensitivity analysis experiments
        sensitivity = ablation_design["sensitivity_analysis"]
        for param_name, param_info in sensitivity["parameters"].items():
            baseline_config = {
                "amc_instance": 1.0,
                "amc_temporal": 0.5,
                "amc_margin": 0.5,
                "temperature_scheduling": True,
                "min_tau": 0.15,
                "max_tau": 0.75,
                "t_max": 10.5
            }
            
            for param_value in param_info["range"]:
                config = baseline_config.copy()
                config[param_name] = param_value
                
                experiments.append({
                    "experiment_type": "sensitivity_analysis",
                    "scenario_name": f"{param_name}_sensitivity_{param_value}",
                    "description": param_info["description"],
                    "config": config,
                    "parameter_under_test": param_name,
                    "parameter_value": param_value,
                    "baseline_value": param_info["baseline"],
                    "datasets": ["Chinatown"],
                    "epochs": 100,
                    "repeats": 5
                })
        
        logger.info(f"Generated {len(experiments)} ablation experiments")
        return experiments
    
    def analyze_ablation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze ablation study results with statistical testing"""
        
        analysis = {
            "summary": {},
            "statistical_tests": {},
            "component_effects": {},
            "interaction_effects": {},
            "sensitivity_analysis": {},
            "recommendations": []
        }
        
        # Group results by experiment type
        results_by_type = defaultdict(list)
        for result in results:
            results_by_type[result.get("experiment_type", "unknown")].append(result)
        
        # Analyze baseline comparisons
        if "baseline_comparison" in results_by_type:
            analysis["component_effects"]["baseline_comparison"] = self._analyze_baseline_comparison(
                results_by_type["baseline_comparison"]
            )
        
        # Analyze AMC components
        if "amc_component" in results_by_type:
            analysis["component_effects"]["amc_components"] = self._analyze_amc_components(
                results_by_type["amc_component"]
            )
        
        # Analyze temperature schedulers
        if "temperature_scheduler" in results_by_type:
            analysis["component_effects"]["temperature_schedulers"] = self._analyze_temperature_schedulers(
                results_by_type["temperature_scheduler"]
            )
        
        # Analyze interactions
        if "amc_temp_interaction" in results_by_type:
            analysis["interaction_effects"]["amc_temperature"] = self._analyze_amc_temp_interactions(
                results_by_type["amc_temp_interaction"]
            )
        
        # Analyze sensitivity
        if "sensitivity_analysis" in results_by_type:
            analysis["sensitivity_analysis"] = self._analyze_parameter_sensitivity(
                results_by_type["sensitivity_analysis"]
            )
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_baseline_comparison(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze baseline comparison results"""
        
        analysis = {
            "scenario_performance": {},
            "statistical_significance": {},
            "effect_sizes": {}
        }
        
        # Group by scenario and dataset
        scenario_data = defaultdict(lambda: defaultdict(list))
        for result in results:
            if result.get("status") == "success" and "metrics" in result:
                scenario = result["scenario_name"]
                dataset = result["dataset"]
                accuracy = result["metrics"].get("accuracy", 0.0)
                scenario_data[scenario][dataset].append(accuracy)
        
        # Calculate statistics for each scenario
        for scenario, dataset_results in scenario_data.items():
            scenario_stats = {}
            for dataset, accuracies in dataset_results.items():
                if len(accuracies) > 0:
                    scenario_stats[dataset] = {
                        "mean": np.mean(accuracies),
                        "std": np.std(accuracies),
                        "count": len(accuracies),
                        "min": np.min(accuracies),
                        "max": np.max(accuracies)
                    }
            analysis["scenario_performance"][scenario] = scenario_stats
        
        # Statistical significance testing
        scenarios = list(scenario_data.keys())
        if len(scenarios) >= 2:
            for i, scenario1 in enumerate(scenarios):
                for scenario2 in scenarios[i+1:]:
                    # Compare scenarios across datasets
                    for dataset in set(scenario_data[scenario1].keys()) & set(scenario_data[scenario2].keys()):
                        data1 = scenario_data[scenario1][dataset]
                        data2 = scenario_data[scenario2][dataset]
                        
                        if len(data1) > 1 and len(data2) > 1:
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            
                            # Calculate effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                                (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                               (len(data1) + len(data2) - 2))
                            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                            
                            comparison_key = f"{scenario1}_vs_{scenario2}_{dataset}"
                            analysis["statistical_significance"][comparison_key] = {
                                "t_statistic": t_stat,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "effect_size": cohens_d,
                                "interpretation": self._interpret_effect_size(cohens_d)
                            }
        
        return analysis
    
    def _analyze_amc_components(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze AMC component effects"""
        
        analysis = {
            "component_comparison": {},
            "parameter_effects": {},
            "best_configurations": {}
        }
        
        # Group by component type
        component_data = defaultdict(list)
        for result in results:
            if result.get("status") == "success" and "metrics" in result:
                component = result["scenario_name"]
                accuracy = result["metrics"].get("accuracy", 0.0)
                auprc = result["metrics"].get("auprc", 0.0)
                component_data[component].append({
                    "accuracy": accuracy,
                    "auprc": auprc,
                    "config": result["config"],
                    "dataset": result["dataset"]
                })
        
        # Analyze each component
        for component, data in component_data.items():
            if len(data) > 0:
                accuracies = [d["accuracy"] for d in data]
                auprcs = [d["auprc"] for d in data]
                
                analysis["component_comparison"][component] = {
                    "accuracy": {
                        "mean": np.mean(accuracies),
                        "std": np.std(accuracies),
                        "count": len(accuracies)
                    },
                    "auprc": {
                        "mean": np.mean(auprcs),
                        "std": np.std(auprcs),
                        "count": len(auprcs)
                    },
                    "best_result": data[np.argmax(accuracies)]
                }
        
        # Find best configurations
        all_results = []
        for component_results in component_data.values():
            all_results.extend(component_results)
        
        if all_results:
            best_accuracy = max(all_results, key=lambda x: x["accuracy"])
            best_auprc = max(all_results, key=lambda x: x["auprc"])
            
            analysis["best_configurations"] = {
                "highest_accuracy": {
                    "accuracy": best_accuracy["accuracy"],
                    "config": best_accuracy["config"],
                    "dataset": best_accuracy["dataset"]
                },
                "highest_auprc": {
                    "auprc": best_auprc["auprc"],
                    "config": best_auprc["config"],
                    "dataset": best_auprc["dataset"]
                }
            }
        
        return analysis
    
    def _analyze_temperature_schedulers(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temperature scheduler effects"""
        
        analysis = {
            "scheduler_comparison": {},
            "convergence_analysis": {},
            "best_schedulers": {}
        }
        
        # Group by scheduler
        scheduler_data = defaultdict(list)
        for result in results:
            if result.get("status") == "success" and "metrics" in result:
                scheduler = result["scenario_name"]
                accuracy = result["metrics"].get("accuracy", 0.0)
                training_time = result["metrics"].get("training_time", 0.0)
                scheduler_data[scheduler].append({
                    "accuracy": accuracy,
                    "training_time": training_time,
                    "dataset": result["dataset"]
                })
        
        # Analyze each scheduler
        for scheduler, data in scheduler_data.items():
            if len(data) > 0:
                accuracies = [d["accuracy"] for d in data]
                times = [d["training_time"] for d in data if d["training_time"] > 0]
                
                analysis["scheduler_comparison"][scheduler] = {
                    "accuracy": {
                        "mean": np.mean(accuracies),
                        "std": np.std(accuracies),
                        "count": len(accuracies)
                    },
                    "training_time": {
                        "mean": np.mean(times) if times else 0.0,
                        "std": np.std(times) if times else 0.0,
                        "count": len(times)
                    },
                    "efficiency": np.mean(accuracies) / np.mean(times) if times and np.mean(times) > 0 else 0.0
                }
        
        return analysis
    
    def _analyze_amc_temp_interactions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze AMC-temperature interactions"""
        
        analysis = {
            "interaction_matrix": {},
            "synergistic_effects": [],
            "antagonistic_effects": []
        }
        
        # Group by AMC level and temperature setting
        interaction_data = {}
        for result in results:
            if result.get("status") == "success" and "metrics" in result:
                config = result["config"]
                amc_level = config.get("amc_instance", 0.0)
                temp_enabled = config.get("temperature_scheduling", False)
                temp_method = config.get("scheduler_method", "none") if temp_enabled else "none"
                
                key = f"amc_{amc_level}_temp_{temp_method}"
                if key not in interaction_data:
                    interaction_data[key] = []
                
                interaction_data[key].append(result["metrics"].get("accuracy", 0.0))
        
        # Calculate interaction effects
        for key, accuracies in interaction_data.items():
            if len(accuracies) > 0:
                analysis["interaction_matrix"][key] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "count": len(accuracies)
                }
        
        return analysis
    
    def _analyze_parameter_sensitivity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter sensitivity"""
        
        analysis = {
            "parameter_effects": {},
            "sensitivity_scores": {},
            "optimal_ranges": {}
        }
        
        # Group by parameter
        param_data = defaultdict(list)
        for result in results:
            if result.get("status") == "success" and "metrics" in result:
                param_name = result["parameter_under_test"]
                param_value = result["parameter_value"]
                accuracy = result["metrics"].get("accuracy", 0.0)
                
                param_data[param_name].append({
                    "value": param_value,
                    "accuracy": accuracy,
                    "baseline": result["baseline_value"]
                })
        
        # Analyze sensitivity for each parameter
        for param_name, data in param_data.items():
            if len(data) > 2:
                values = [d["value"] for d in data]
                accuracies = [d["accuracy"] for d in data]
                baseline = data[0]["baseline"]
                
                # Calculate sensitivity metrics
                value_range = max(values) - min(values)
                accuracy_range = max(accuracies) - min(accuracies)
                sensitivity = accuracy_range / value_range if value_range > 0 else 0
                
                # Find optimal range
                sorted_data = sorted(data, key=lambda x: x["accuracy"], reverse=True)
                top_20_percent = sorted_data[:max(1, len(sorted_data) // 5)]
                optimal_values = [d["value"] for d in top_20_percent]
                
                analysis["parameter_effects"][param_name] = {
                    "sensitivity_score": sensitivity,
                    "accuracy_range": accuracy_range,
                    "baseline_value": baseline,
                    "best_value": sorted_data[0]["value"],
                    "best_accuracy": sorted_data[0]["accuracy"],
                    "optimal_range": [min(optimal_values), max(optimal_values)]
                }
        
        return analysis
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on ablation analysis"""
        
        recommendations = []
        
        # Component recommendations
        if "component_effects" in analysis:
            component_effects = analysis["component_effects"]
            
            if "baseline_comparison" in component_effects:
                baseline_comp = component_effects["baseline_comparison"]
                # Find best performing scenario
                best_scenario = None
                best_performance = 0
                
                for scenario, perf in baseline_comp["scenario_performance"].items():
                    avg_perf = np.mean([dataset_stats["mean"] for dataset_stats in perf.values()])
                    if avg_perf > best_performance:
                        best_performance = avg_perf
                        best_scenario = scenario
                
                if best_scenario:
                    recommendations.append(f"Use {best_scenario} configuration for best overall performance")
            
            if "amc_components" in component_effects:
                amc_comp = component_effects["amc_components"]
                if "best_configurations" in amc_comp:
                    best_config = amc_comp["best_configurations"]["highest_accuracy"]
                    recommendations.append(f"For highest accuracy, use AMC configuration: {best_config['config']}")
        
        # Sensitivity recommendations
        if "sensitivity_analysis" in analysis:
            sensitivity = analysis["sensitivity_analysis"]["parameter_effects"]
            
            for param_name, param_analysis in sensitivity.items():
                if param_analysis["sensitivity_score"] > 0.1:  # High sensitivity
                    optimal_range = param_analysis["optimal_range"]
                    recommendations.append(
                        f"{param_name} is highly sensitive - use values in range {optimal_range[0]:.3f} to {optimal_range[1]:.3f}"
                    )
                elif param_analysis["sensitivity_score"] < 0.01:  # Low sensitivity
                    recommendations.append(
                        f"{param_name} has low sensitivity - can use default value {param_analysis['baseline_value']}"
                    )
        
        return recommendations
    
    def export_ablation_report(self, 
                             analysis: Dict[str, Any], 
                             output_file: str = None) -> str:
        """Export comprehensive ablation analysis report"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.ablation_results_dir / f"ablation_analysis_report_{timestamp}.json")
        
        # Create comprehensive report
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "analysis_framework": "TimeHUT Ablation Framework",
                "version": "1.0"
            },
            "analysis": analysis,
            "summary": self._create_executive_summary(analysis)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Ablation analysis report saved to: {output_file}")
        return output_file
    
    def _create_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of ablation results"""
        
        summary = {
            "key_findings": [],
            "best_configurations": {},
            "component_importance": {},
            "recommendations": analysis.get("recommendations", [])
        }
        
        # Extract key findings
        if "component_effects" in analysis:
            if "baseline_comparison" in analysis["component_effects"]:
                summary["key_findings"].append(
                    "Baseline comparison shows significant differences between TimeHUT variants"
                )
            
            if "amc_components" in analysis["component_effects"]:
                summary["key_findings"].append(
                    "AMC components contribute differentially to performance"
                )
            
            if "temperature_schedulers" in analysis["component_effects"]:
                summary["key_findings"].append(
                    "Temperature scheduling methods show varying effectiveness"
                )
        
        return summary


def main():
    """Main function for running ablation studies"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TimeHUT Ablation Studies Framework")
    parser.add_argument('--design', action='store_true', 
                       help='Design ablation experiments')
    parser.add_argument('--analyze', type=str,
                       help='Analyze results from file')
    parser.add_argument('--base-dir', type=str, default='/home/amin/TSlib',
                       help='Base directory')
    
    args = parser.parse_args()
    
    framework = TimeHUTAblationFramework(base_dir=args.base_dir)
    
    if args.design:
        # Design ablation experiments
        ablation_design = framework.design_component_ablations()
        experiments = framework.generate_ablation_experiments(ablation_design)
        
        # Save experiment design
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/home/amin/TSlib/models/timehut/analysis_results/ablations/ablation_design_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "ablation_design": ablation_design,
                "experiments": experiments
            }, f, indent=2)
        
        print(f"✅ Ablation design created: {len(experiments)} experiments")
        print(f"📁 Saved to: {output_file}")
    
    if args.analyze:
        # Analyze results
        with open(args.analyze, 'r') as f:
            results = json.load(f)
        
        analysis = framework.analyze_ablation_results(results)
        report_file = framework.export_ablation_report(analysis)
        
        print(f"✅ Analysis completed")
        print(f"📊 Report saved to: {report_file}")


if __name__ == "__main__":
    main()
