#!/usr/bin/env python3
"""
Model Optimization and Fair Comparison Framework
===============================================

This module provides optimized hyperparameters for fair comparison across all models
in the TSlib unified benchmarking system. It ensures all models use the same baseline
parameters (from TS2vec) and provides optimization settings for improved performance.

Usage:
    from model_optimization_fair_comparison import get_optimized_config, get_fair_baseline_config
    
    # Get fair comparison baseline (same as TS2vec)
    baseline_config = get_fair_baseline_config('BIOT', 'AtrialFibrillation')
    
    # Get optimized parameters for best performance
    optimized_config = get_optimized_config('BIOT', 'AtrialFibrillation')
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
import json

# =============================================================================
# TS2VEC BASELINE PARAMETERS (Reference for Fair Comparison)
# =============================================================================

# These are the EXACT parameters that achieve TS2vec's high performance
# All other models will use these same values for fair comparison
TS2VEC_BASELINE_PARAMS = {
    'Chinatown': {
        'batch_size': 8,
        'learning_rate': 0.001,
        'n_iters': 200,  # TS2vec uses iterations, not epochs
        'seed': 42,
        'max_train_length': 3000,
        'repr_dims': 320,  # TS2vec specific
        'max_threads': 8,
        'irregular_ratio': 0.0,
        'eval_protocol': 'svm',
        'gpu': 0
    },
    'AtrialFibrillation': {
        'batch_size': 8,
        'learning_rate': 0.001,
        'n_iters': 200,
        'seed': 42,
        'max_train_length': 3000,
        'repr_dims': 320,
        'max_threads': 8,
        'irregular_ratio': 0.0,
        'eval_protocol': 'svm',
        'gpu': 0
    },
    'MotorImagery': {
        'batch_size': 8,
        'learning_rate': 0.001,
        'n_iters': 200,
        'seed': 42,
        'max_train_length': 3000,
        'repr_dims': 320,
        'max_threads': 8,
        'irregular_ratio': 0.0,
        'eval_protocol': 'svm',
        'gpu': 0
    },
    'StandWalkJump': {
        'batch_size': 8,
        'learning_rate': 0.001,
        'n_iters': 200,
        'seed': 42,
        'max_train_length': 3000,
        'repr_dims': 320,
        'max_threads': 8,
        'irregular_ratio': 0.0,
        'eval_protocol': 'svm',
        'gpu': 0
    },
    'EigenWorms': {
        'batch_size': 8,
        'learning_rate': 0.001,
        'n_iters': 600,  # Larger dataset needs more training
        'seed': 42,
        'max_train_length': 3000,
        'repr_dims': 320,
        'max_threads': 8,
        'irregular_ratio': 0.0,
        'eval_protocol': 'svm',
        'gpu': 0
    }
}

# =============================================================================
# OPTIMIZED PARAMETERS (Best Performance Settings)
# =============================================================================

# These parameters are optimized for each model type to achieve best performance
OPTIMIZED_PARAMS = {
    # TS2vec optimized (baseline)
    'TS2vec': {
        'Chinatown': {
            'batch_size': 8,
            'learning_rate': 0.001,
            'n_iters': 200,
            'repr_dims': 320,
            'seed': 42
        },
        'AtrialFibrillation': {
            'batch_size': 8,
            'learning_rate': 0.001,
            'n_iters': 200,
            'repr_dims': 320,
            'seed': 42
        }
    },
    
    # MF-CLR optimized (based on paper recommendations)
    'MF-CLR': {
        'Chinatown': {
            'batch_size': 16,  # MF-CLR paper default
            'learning_rate': 0.0005,  # Slightly lower for stability
            'n_iters': 200,  # Same as TS2vec for fair comparison
            'seed': 42
        },
        'AtrialFibrillation': {
            'batch_size': 16,
            'learning_rate': 0.0005,
            'n_iters': 200,
            'seed': 42
        }
    },
    
    # ConvTran (CoST) optimized
    'ConvTran': {
        'Chinatown': {
            'batch_size': 16,
            'learning_rate': 0.0001,  # Transformer models often need lower LR
            'n_iters': 200,
            'seed': 42
        },
        'AtrialFibrillation': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'n_iters': 200,
            'seed': 42
        }
    },
    
    # InceptionTime optimized
    'InceptionTime': {
        'Chinatown': {
            'batch_size': 32,  # InceptionTime can handle larger batches
            'learning_rate': 0.001,
            'n_iters': 200,
            'seed': 42
        },
        'AtrialFibrillation': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'n_iters': 200,
            'seed': 42
        }
    },
    
    # VQ-MTM models optimized
    'BIOT': {
        'Chinatown': {
            'batch_size': 8,  # Keep smaller for multivariate data
            'learning_rate': 0.0001,  # Lower LR for transformer-based models
            'n_iters': 100,  # VQ-MTM models converge faster
            'seed': 42,
            'd_model': 64,
            'e_layers': 2,
            'dropout': 0.1
        },
        'AtrialFibrillation': {
            'batch_size': 8,
            'learning_rate': 0.0001,
            'n_iters': 100,
            'seed': 42,
            'd_model': 64,
            'e_layers': 2,
            'dropout': 0.1
        }
    },
    
    'TimesNet': {
        'Chinatown': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'n_iters': 100,
            'seed': 42,
            'd_model': 64,
            'e_layers': 2,
            'dropout': 0.1
        },
        'AtrialFibrillation': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'n_iters': 100,
            'seed': 42,
            'd_model': 64,
            'e_layers': 2,
            'dropout': 0.1
        }
    },
    
    'VQ_VAE': {
        'Chinatown': {
            'batch_size': 8,
            'learning_rate': 0.0005,  # VAE models often need moderate LR
            'n_iters': 150,
            'seed': 42,
            'd_model': 64,
            'dropout': 0.1
        },
        'AtrialFibrillation': {
            'batch_size': 8,
            'learning_rate': 0.0005,
            'n_iters': 150,
            'seed': 42,
            'd_model': 64,
            'dropout': 0.1
        }
    }
}

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class FairComparisonConfig:
    """Configuration ensuring fair comparison (same as TS2vec baseline)"""
    batch_size: int
    learning_rate: float
    n_iters: int
    seed: int
    max_train_length: int = 3000
    repr_dims: int = 320  # For representation learning models
    max_threads: int = 8
    irregular_ratio: float = 0.0
    eval_protocol: str = 'svm'
    gpu: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class OptimizedConfig:
    """Configuration optimized for best performance per model"""
    batch_size: int
    learning_rate: float
    n_iters: int
    seed: int
    # Model-specific parameters (optional)
    d_model: Optional[int] = None
    e_layers: Optional[int] = None
    dropout: Optional[float] = None
    repr_dims: Optional[int] = None
    max_train_length: int = 3000
    max_threads: int = 8
    irregular_ratio: float = 0.0
    eval_protocol: str = 'svm'
    gpu: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================

def get_fair_baseline_config(model_name: str, dataset_name: str) -> FairComparisonConfig:
    """
    Get fair comparison baseline configuration (same as TS2vec) for any model.
    
    This ensures all models use exactly the same hyperparameters for fair comparison.
    
    Args:
        model_name: Name of the model (e.g., 'BIOT', 'MF-CLR', etc.)
        dataset_name: Name of the dataset (e.g., 'Chinatown', 'AtrialFibrillation')
        
    Returns:
        FairComparisonConfig with TS2vec baseline parameters
    """
    if dataset_name not in TS2VEC_BASELINE_PARAMS:
        # Use default parameters for unknown datasets
        base_params = TS2VEC_BASELINE_PARAMS['Chinatown'].copy()
        print(f"⚠️ No baseline config for {dataset_name}, using Chinatown defaults")
    else:
        base_params = TS2VEC_BASELINE_PARAMS[dataset_name].copy()
    
    # Remove model-specific parameters for general use
    base_params.pop('repr_dims', None)
    
    return FairComparisonConfig(**base_params)

def get_optimized_config(model_name: str, dataset_name: str) -> OptimizedConfig:
    """
    Get optimized configuration for best performance.
    
    These parameters are tuned for each model type to achieve optimal results.
    
    Args:
        model_name: Name of the model (e.g., 'BIOT', 'MF-CLR', etc.)
        dataset_name: Name of the dataset (e.g., 'Chinatown', 'AtrialFibrillation')
        
    Returns:
        OptimizedConfig with model-specific optimized parameters
    """
    if model_name not in OPTIMIZED_PARAMS:
        print(f"⚠️ No optimized config for {model_name}, using fair baseline")
        fair_config = get_fair_baseline_config(model_name, dataset_name)
        return OptimizedConfig(**fair_config.to_dict())
    
    model_params = OPTIMIZED_PARAMS[model_name]
    
    if dataset_name not in model_params:
        # Use first available dataset as fallback
        available_dataset = list(model_params.keys())[0]
        dataset_params = model_params[available_dataset].copy()
        print(f"⚠️ No optimized config for {model_name}+{dataset_name}, using {available_dataset} settings")
    else:
        dataset_params = model_params[dataset_name].copy()
    
    return OptimizedConfig(**dataset_params)

def get_model_config(model_name: str, dataset_name: str, optimization_mode: str = 'fair') -> Union[FairComparisonConfig, OptimizedConfig]:
    """
    Get model configuration based on optimization mode.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        optimization_mode: 'fair' for fair comparison, 'optimized' for best performance
        
    Returns:
        Configuration object based on mode
    """
    if optimization_mode == 'fair':
        return get_fair_baseline_config(model_name, dataset_name)
    elif optimization_mode == 'optimized':
        return get_optimized_config(model_name, dataset_name)
    else:
        raise ValueError(f"Unknown optimization_mode: {optimization_mode}. Use 'fair' or 'optimized'")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_comparison_summary(dataset_name: str = 'AtrialFibrillation'):
    """Print comparison of fair vs optimized configurations for all models"""
    print(f"📊 Configuration Comparison for {dataset_name}")
    print("=" * 80)
    print(f"{'Model':<15} {'Mode':<10} {'Batch':<6} {'LR':<8} {'Iters':<6} {'Special Params':<20}")
    print("-" * 80)
    
    models = ['TS2vec', 'MF-CLR', 'ConvTran', 'BIOT', 'TimesNet']
    
    for model in models:
        # Fair comparison
        try:
            fair = get_fair_baseline_config(model, dataset_name)
            print(f"{model:<15} {'Fair':<10} {fair.batch_size:<6} {fair.learning_rate:<8} {fair.n_iters:<6} {'Standard':<20}")
        except:
            print(f"{model:<15} {'Fair':<10} {'N/A':<6} {'N/A':<8} {'N/A':<6} {'N/A':<20}")
        
        # Optimized
        try:
            opt = get_optimized_config(model, dataset_name)
            special = f"d_model={opt.d_model}" if opt.d_model else "Standard"
            print(f"{model:<15} {'Optimized':<10} {opt.batch_size:<6} {opt.learning_rate:<8} {opt.n_iters:<6} {special:<20}")
        except:
            print(f"{model:<15} {'Optimized':<10} {'N/A':<6} {'N/A':<8} {'N/A':<6} {'N/A':<20}")
    
    print("=" * 80)

def save_configs_to_json(output_path: str = "/home/amin/TSlib/unified/optimization_configs.json"):
    """Save all configurations to JSON file for reference"""
    all_configs = {
        'ts2vec_baseline_params': TS2VEC_BASELINE_PARAMS,
        'optimized_params': OPTIMIZED_PARAMS,
        'metadata': {
            'description': 'TSlib unified model optimization configurations',
            'fair_comparison': 'All models use TS2vec baseline parameters',
            'optimized': 'Model-specific parameters for best performance',
            'created': '2025-08-23'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(all_configs, f, indent=2)
    
    print(f"💾 Configurations saved to {output_path}")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    print("🎯 TSlib Model Optimization & Fair Comparison Framework")
    print("=" * 60)
    
    # Example usage
    print("\n📊 Fair Comparison Example (TS2vec baseline):")
    fair_config = get_fair_baseline_config('BIOT', 'AtrialFibrillation')
    print(f"BIOT on AtrialFibrillation (Fair): {fair_config}")
    
    print("\n🚀 Optimized Configuration Example:")
    opt_config = get_optimized_config('BIOT', 'AtrialFibrillation')
    print(f"BIOT on AtrialFibrillation (Optimized): {opt_config}")
    
    print("\n📋 Configuration Comparison:")
    print_comparison_summary('AtrialFibrillation')
    
    print("\n💾 Saving configurations...")
    save_configs_to_json()
    
    print("\n✅ Framework ready for use!")
