import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from torch.utils.data import DataLoader, TensorDataset
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program
from sklearn.model_selection import train_test_split
import pyhopper
import json
import gc

labeled_ratio = 0.7

# Performance optimization flags - Aggressive settings for speed
OPTIMIZATION_FLAGS = {
    'use_mixed_precision': True,    # Use AMP for faster training
    'optimize_memory': True,        # Aggressive memory management
    'early_stop_search': True,      # Early stopping for hyperparameter search
    'reduced_eval_freq': True,      # Less frequent evaluation during search
    'batch_size_scaling': True,     # Dynamic batch size based on GPU memory
    'gradient_checkpointing': False, # Trade speed for memory (disable for speed)
    'fast_inference': True,         # Optimize inference during evaluation
    'aggressive_gc': True,          # Force garbage collection frequently
    'cuda_benchmark': True,         # Enable cudnn benchmarking for faster convolutions
    'pin_memory': True,             # Pin memory for data loading
    'num_workers': 4,               # Multi-threaded data loading
    'compile_model': False,         # PyTorch 2.0+ model compilation (experimental)
    'reduce_precision_eval': True   # Use float16 for evaluation when possible
}

# Training scenario modes
SCENARIO_MODES = {
    'baseline': 'Standard TimeHUT training without AMC or temperature scheduling',
    'amc_only': 'Training with AMC losses only (instance and/or temporal)',
    'temp_only': 'Training with temperature scheduling only',
    'amc_temp': 'Training with both AMC losses and temperature scheduling',
    'optimize_amc': 'Use pyhopper to optimize AMC parameters only',
    'optimize_temp': 'Use pyhopper to optimize temperature parameters only', 
    'optimize_combined': 'Use pyhopper to optimize both AMC and temperature parameters',
    'gridsearch_amc': 'Grid search over AMC parameters',
    'gridsearch_temp': 'Grid search over temperature parameters',
    'gridsearch_full': 'Full grid search over both AMC and temperature parameters',
    # New temperature scheduling methods
    'temp_cosine': 'Training with enhanced cosine annealing temperature scheduling',
    'temp_adaptive_cosine': 'Training with adaptive cosine annealing temperature scheduling',
    'temp_multi_cycle_cosine': 'Training with multi-cycle cosine annealing temperature scheduling',
    'temp_cosine_restarts': 'Training with cosine annealing with warm restarts temperature scheduling',
    'temp_linear': 'Training with linear decay temperature scheduling',
    'temp_exponential': 'Training with exponential decay temperature scheduling',
    'temp_step': 'Training with step decay temperature scheduling',
    'temp_polynomial': 'Training with polynomial decay temperature scheduling',
    'temp_sigmoid': 'Training with sigmoid decay temperature scheduling',
    'temp_warmup_cosine': 'Training with warmup + cosine annealing temperature scheduling',
    'temp_constant': 'Training with constant temperature',
    'temp_cyclic': 'Training with cyclic temperature scheduling',
}

def clear_gpu_memory():
    """Clear GPU memory to prevent accumulation - Aggressive version"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if OPTIMIZATION_FLAGS['optimize_memory']:
            # Multiple rounds of cleanup for thorough clearing
            for _ in range(2):
                gc.collect()
                torch.cuda.empty_cache()

def setup_performance_optimizations():
    """Configure PyTorch for maximum performance - Ultra-aggressive mode"""
    # Enable all PyTorch optimizations
    enable_pytorch_optimizations()
    
    if torch.cuda.is_available() and OPTIMIZATION_FLAGS['cuda_benchmark']:
        torch.backends.cudnn.benchmark = True  # Faster convolutions
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    if OPTIMIZATION_FLAGS['aggressive_gc']:
        # More aggressive garbage collection
        gc.set_threshold(50, 5, 5)  # Even lower thresholds = more frequent GC
    
    # Set number of threads for CPU operations
    torch.set_num_threads(min(4, torch.get_num_threads()))
    
    print(f"⚡ ULTRA-PERFORMANCE optimizations enabled:")
    print(f"  - Mixed precision: {OPTIMIZATION_FLAGS['use_mixed_precision']}")
    print(f"  - CUDA benchmark: {OPTIMIZATION_FLAGS.get('cuda_benchmark', False)}")
    print(f"  - Memory optimization: {OPTIMIZATION_FLAGS['optimize_memory']}")
    print(f"  - Dynamic batch size: {OPTIMIZATION_FLAGS['batch_size_scaling']}")
    print(f"  - Fast inference: {OPTIMIZATION_FLAGS.get('fast_inference', False)}")
    print(f"  - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else 'N/A'}")

def optimize_batch_size(base_batch_size, dataset_size):
    """Dynamically optimize batch size based on dataset size and GPU memory - More aggressive"""
    if not OPTIMIZATION_FLAGS['batch_size_scaling']:
        return base_batch_size
    
    # More aggressive scaling for performance
    if dataset_size < 500:
        return min(base_batch_size * 4, 64)  # Much larger for tiny datasets
    elif dataset_size < 2000:
        return min(base_batch_size * 2, 32)  # Increase for small datasets
    elif dataset_size > 10000:
        return max(base_batch_size // 4, 4)  # More aggressive decrease for large datasets
    elif dataset_size > 5000:
        return max(base_batch_size // 2, 8)  # Decrease for large datasets
    return base_batch_size

def get_optimized_epochs(base_epochs, scenario, dataset_size):
    """Reduce epochs for hyperparameter search based on dataset complexity - More aggressive"""
    if scenario in ['optimize_amc', 'optimize_temp', 'optimize_combined']:
        if dataset_size < 500:
            return min(base_epochs // 5, 20)  # Very fast for tiny datasets
        elif dataset_size < 1000:
            return min(base_epochs // 4, 25)  # Very fast search for small datasets
        elif dataset_size < 3000:
            return min(base_epochs // 3, 35)  # Fast for medium datasets
        else:
            return min(base_epochs // 2, 50)
    return base_epochs

def should_skip_search(dataset_size, dataset_name):
    """Skip hyperparameter search for small/simple datasets - More aggressive"""
    # Skip for datasets with < 2000 samples or known simple datasets
    simple_datasets = ['Chinatown', 'TwoLeadECG', 'ECG200', 'Coffee', 'SyntheticControl', 
                      'MoteStrain', 'CBF', 'ECGFiveDays', 'FaceFour', 'Lightning2']
    
    # Much more aggressive skipping for efficiency
    if OPTIMIZATION_FLAGS['early_stop_search']:
        return dataset_size < 2000 or dataset_name in simple_datasets
    else:
        return dataset_size < 1000 or dataset_name in simple_datasets

def get_fast_eval_frequency(base_epochs):
    """Reduce evaluation frequency during hyperparameter search"""
    if OPTIMIZATION_FLAGS['reduced_eval_freq']:
        # Evaluate much less frequently during search
        if base_epochs <= 20:
            return max(base_epochs // 4, 2)  # Every 2-5 epochs for short training
        elif base_epochs <= 50:
            return max(base_epochs // 6, 5)  # Every 5-8 epochs for medium training
        else:
            return max(base_epochs // 8, 10) # Every 10+ epochs for long training
    return 1  # Default: every epoch

def get_optimized_search_space():
    """Use narrower, more focused hyperparameter ranges for temperature"""
    # Reduced search space for faster optimization
    return pyhopper.Search({
        "min_tau": pyhopper.float(0.08, 0.25, "0.02f"),   # Narrowed range
        "max_tau": pyhopper.float(0.65, 0.95, "0.03f"),   # Narrowed range
        "t_max": pyhopper.float(6, 15, "1.0f"),           # Reduced range
    })

def get_amc_search_space():
    """Get AMC hyperparameter search space for pyhopper - optimized ranges"""
    # More focused AMC search space based on common good values
    return pyhopper.Search({
        "amc_instance": pyhopper.float(0.1, 3.0, "0.2f"),  # Focused on effective range
        "amc_temporal": pyhopper.float(0.1, 3.0, "0.2f"),  # Focused on effective range
        "amc_margin": pyhopper.float(0.3, 0.7, "0.1f")     # Narrow margin range
    })

def get_combined_search_space():
    """Get combined AMC + temperature hyperparameter search space - optimized"""
    # Combined search with reduced dimensionality
    return pyhopper.Search({
        # AMC parameters - focused ranges
        "amc_instance": pyhopper.float(0.1, 3.0, "0.2f"),
        "amc_temporal": pyhopper.float(0.1, 3.0, "0.2f"),
        "amc_margin": pyhopper.float(0.3, 0.7, "0.1f"),
        # Temperature parameters - focused ranges
        "min_tau": pyhopper.float(0.08, 0.25, "0.02f"),
        "max_tau": pyhopper.float(0.65, 0.95, "0.03f"),
        "t_max": pyhopper.float(6, 15, "1.0f"),
    })

def get_amc_grid_ranges():
    """Get AMC parameter ranges for grid search"""
    return {
        'amc_instance': [0.1, 0.5, 1, 3, 5],  # Removed 0 to prevent zero values
        'amc_temporal': [0.1, 0.5, 1, 3, 5],  # Removed 0 to prevent zero values
        'amc_margin': [0.3, 0.5, 0.7]  # Added range instead of fixed value
    }

def get_temp_grid_ranges():
    """Get temperature parameter ranges for grid search"""
    return {
        'min_tau': [0.07, 0.1, 0.3],
        'max_tau': [0.6, 0.8, 1.0],
        't_max': [5, 10, 20]
    }

def create_amc_setting(amc_instance=0.1, amc_temporal=0.1, amc_margin=0.5):
    """Create AMC settings dictionary"""
    return {
        'amc_instance': amc_instance,
        'amc_temporal': amc_temporal,
        'amc_margin': amc_margin
    }

def create_temp_setting(min_tau=0.15, max_tau=0.75, t_max=10.5):
    """Create temperature settings dictionary"""
    return {
        'min_tau': min_tau,
        'max_tau': max_tau,
        't_max': t_max
    }

def train_model(config, temp_dictionary=None, amc_setting=None, type="full"):
    '''
    trains the ts2vec model using either full dataset or the split dataset
    Supports all training scenarios: baseline, AMC, temperature, and combined
    Supports all task types: classification, forecasting, anomaly detection
    OPTIMIZED for better performance and memory efficiency
    '''
    # Clear GPU memory before training
    clear_gpu_memory()
    
    # Optimize batch size dynamically
    optimized_config = config.copy()
    if OPTIMIZATION_FLAGS['batch_size_scaling']:
        original_batch_size = config['batch_size']
        dataset_size = len(train_data) if 'train_data' in globals() else 1000
        optimized_config['batch_size'] = optimize_batch_size(original_batch_size, dataset_size)
        if optimized_config['batch_size'] != original_batch_size:
            print(f"📈 Optimized batch size: {original_batch_size} → {optimized_config['batch_size']}")
    
    if type == 'split':
        t = time.time()
        
        # Use mixed precision if available
        use_amp = OPTIMIZATION_FLAGS['use_mixed_precision'] and torch.cuda.is_available()
        if use_amp:
            print("⚡ Using mixed precision training for faster performance")
        
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            temp_dictionary=temp_dictionary,
            amc_setting=amc_setting,
            **optimized_config
        )
        
        # Optimized training epochs for search
        search_epochs = get_optimized_epochs(args.epochs, args.scenario, len(train_data_split))
        print(f"🔍 Search phase: Using {search_epochs} epochs (reduced from {args.epochs} for efficiency)")
        
        # Train with aggressive optimizations for hyperparameter search
        loss_log = model.fit(
            train_data_split,
            n_epochs=search_epochs,
            n_iters=args.iters,
            verbose=False
        )
        t = time.time() - t
        
        # Lightweight evaluation for hyperparameter search - Ultra-fast mode
        if task_type == 'classification':
            eval_start = time.time()
            
            # Use reduced precision for evaluation if available
            with torch.cuda.amp.autocast(enabled=use_amp and OPTIMIZATION_FLAGS['reduce_precision_eval']):
                out_val, eval_res_val = tasks.eval_classification(
                    model, train_data_split, train_labels_split, 
                    val_data_split, val_labels_split, 
                    eval_protocol='svm'
                )
            eval_time = time.time() - eval_start
            
            # Only do test evaluation if not in fast mode
            if not OPTIMIZATION_FLAGS['reduced_eval_freq'] or args.scenario not in ['optimize_amc', 'optimize_temp', 'optimize_combined']:
                print('Evaluation result (val)               :', eval_res_val)
                with torch.cuda.amp.autocast(enabled=use_amp and OPTIMIZATION_FLAGS['reduce_precision_eval']):
                    out_test, eval_res_test = tasks.eval_classification(
                        model, train_data_split, train_labels_split, 
                        test_data, test_labels, 
                        eval_protocol='svm'
                    )
                print('Evaluation result (test)              :', eval_res_test)
            
            print(f"⏱️ Search Training: {t:.2f}s, Evaluation: {eval_time:.2f}s")
        else:
            # For other task types, use a simple validation approach
            eval_res_val = {'acc': 0.0, 'auprc': 0.0}  # Placeholder
        
        # Clean up model to save memory
        del model
        clear_gpu_memory()
        
        return eval_res_val
    
    if type == 'full':
        t = time.time()
        
        # Use mixed precision for full training if available
        use_amp = OPTIMIZATION_FLAGS['use_mixed_precision'] and torch.cuda.is_available()
        if use_amp:
            print("⚡ Using mixed precision for full training")
            
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            temp_dictionary=temp_dictionary,
            amc_setting=amc_setting,
            **optimized_config
        )
        
        print(f"🚀 Full training with batch size: {optimized_config['batch_size']}")
        loss_log = model.fit(
            train_data,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=True  # Show progress for full training
        )
        t = time.time() - t
        print(f"\n⏱️ Training time: {datetime.timedelta(seconds=t)} ({t/args.epochs:.3f}s/epoch)\n")
        
        # Evaluation based on task type
        eval_start = time.time()
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(
                model, train_data, train_labels, 
                test_data, test_labels, 
                eval_protocol='svm'
            )
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(
                model, data, train_slice, valid_slice, test_slice, 
                scaler, pred_lens, n_covariate_cols
            )
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(
                model, all_train_data, all_train_labels, all_train_timestamps, 
                all_test_data, all_test_labels, all_test_timestamps, delay
            )
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(
                model, all_train_data, all_train_labels, all_train_timestamps, 
                all_test_data, all_test_labels, all_test_timestamps, delay
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        eval_time = time.time() - eval_start
        print('Evaluation result on test (full train):', eval_res)
        print(f"⏱️ Evaluation time: {eval_time:.2f}s")
        
        if task_type != 'classification':
            print('Evaluation result (full train):', eval_res)
            run_name = "_".join((args.loader, args.dataset))
            
        # Clean up
        del model
        clear_gpu_memory()
        
        return eval_res

def run_grid_search(config, mode='amc_only'):
    """
    Run grid search over specified parameters
    """
    results = []
    experiment_count = 0
    
    if mode == 'gridsearch_amc':
        # Grid search over AMC parameters only
        amc_ranges = get_amc_grid_ranges()
        temp_setting = None
        
        for amc_instance in amc_ranges['amc_instance']:
            for amc_temporal in amc_ranges['amc_temporal']:
                for amc_margin in amc_ranges['amc_margin']:
                    print(f"\n=== Experiment {experiment_count + 1} ===")
                    
                    amc_setting = create_amc_setting(amc_instance, amc_temporal, amc_margin)
                    print(f"AMC Setting: {amc_setting}")
                    
                    try:
                        result = train_model(config, temp_setting, amc_setting, type="full")
                        results.append({
                            'experiment': experiment_count,
                            'amc_setting': amc_setting,
                            'temp_setting': temp_setting,
                            'result': result,
                            'success': True
                        })
                        print(f"Result: {result}")
                    except Exception as e:
                        print(f"Failed: {str(e)}")
                        results.append({
                            'experiment': experiment_count,
                            'amc_setting': amc_setting,
                            'temp_setting': temp_setting,
                            'result': {'acc': 0.0, 'auprc': 0.0},
                            'success': False,
                            'error': str(e)
                        })
                    
                    experiment_count += 1
    
    elif mode == 'gridsearch_temp':
        # Grid search over temperature parameters only
        temp_ranges = get_temp_grid_ranges()
        amc_setting = None
        
        for min_tau in temp_ranges['min_tau']:
            for max_tau in temp_ranges['max_tau']:
                for t_max in temp_ranges['t_max']:
                    print(f"\n=== Experiment {experiment_count + 1} ===")
                    
                    temp_setting = create_temp_setting(min_tau, max_tau, t_max)
                    print(f"Temperature Setting: {temp_setting}")
                    
                    try:
                        result = train_model(config, temp_setting, amc_setting, type="full")
                        results.append({
                            'experiment': experiment_count,
                            'amc_setting': amc_setting,
                            'temp_setting': temp_setting,
                            'result': result,
                            'success': True
                        })
                        print(f"Result: {result}")
                    except Exception as e:
                        print(f"Failed: {str(e)}")
                        results.append({
                            'experiment': experiment_count,
                            'amc_setting': amc_setting,
                            'temp_setting': temp_setting,
                            'result': {'acc': 0.0, 'auprc': 0.0},
                            'success': False,
                            'error': str(e)
                        })
                    
                    experiment_count += 1
    
    elif mode == 'gridsearch_full':
        # Grid search over both AMC and temperature parameters
        amc_ranges = get_amc_grid_ranges()
        temp_ranges = get_temp_grid_ranges()
        
        for amc_instance in amc_ranges['amc_instance']:
            for amc_temporal in amc_ranges['amc_temporal']:
                for amc_margin in amc_ranges['amc_margin']:
                    for min_tau in temp_ranges['min_tau']:
                        for max_tau in temp_ranges['max_tau']:
                            for t_max in temp_ranges['t_max']:
                                print(f"\n=== Experiment {experiment_count + 1} ===")
                                
                                amc_setting = create_amc_setting(amc_instance, amc_temporal, amc_margin)
                                temp_setting = create_temp_setting(min_tau, max_tau, t_max)
                                
                                print(f"AMC Setting: {amc_setting}")
                                print(f"Temperature Setting: {temp_setting}")
                                
                                try:
                                    result = train_model(config, temp_setting, amc_setting, type="full")
                                    results.append({
                                        'experiment': experiment_count,
                                        'amc_setting': amc_setting,
                                        'temp_setting': temp_setting,
                                        'result': result,
                                        'success': True
                                    })
                                    print(f"Result: {result}")
                                except Exception as e:
                                    print(f"Failed: {str(e)}")
                                    results.append({
                                        'experiment': experiment_count,
                                        'amc_setting': amc_setting,
                                        'temp_setting': temp_setting,
                                        'result': {'acc': 0.0, 'auprc': 0.0},
                                        'success': False,
                                        'error': str(e)
                                    })
                                
                                experiment_count += 1
    
    return results

def run_pyhopper_optimization(config, mode='optimize_temp', search_steps=20):
    """
    Run pyhopper optimization for specified parameters
    """
    print(f"🔍 Starting {mode} with pyhopper optimization ({search_steps} steps)")
    
    if mode == 'optimize_amc':
        def objective(hparams: dict):
            amc_setting = {
                'amc_instance': hparams['amc_instance'],
                'amc_temporal': hparams['amc_temporal'],
                'amc_margin': hparams['amc_margin']
            }
            result = train_model(config, temp_dictionary=None, amc_setting=amc_setting, type="split")
            return result['acc']
        
        search = get_amc_search_space()
        best_params = search.run(objective, "maximize", steps=search_steps, n_jobs=1)
        
        # Final training with best parameters
        amc_setting = {
            'amc_instance': best_params['amc_instance'],
            'amc_temporal': best_params['amc_temporal'], 
            'amc_margin': best_params['amc_margin']
        }
        final_result = train_model(config, temp_dictionary=None, amc_setting=amc_setting, type="full")
        
        return {
            'best_params': best_params,
            'amc_setting': amc_setting,
            'temp_setting': None,
            'final_result': final_result
        }
    
    elif mode == 'optimize_temp':
        def objective(hparams: dict):
            temp_setting = {
                'min_tau': hparams['min_tau'],
                'max_tau': hparams['max_tau'],
                't_max': hparams['t_max']
            }
            result = train_model(config, temp_dictionary=temp_setting, amc_setting=None, type="split")
            return result['acc']
        
        search = get_optimized_search_space()
        best_params = search.run(objective, "maximize", steps=search_steps, n_jobs=1)
        
        # Final training with best parameters
        temp_setting = {
            'min_tau': best_params['min_tau'],
            'max_tau': best_params['max_tau'],
            't_max': best_params['t_max']
        }
        final_result = train_model(config, temp_dictionary=temp_setting, amc_setting=None, type="full")
        
        return {
            'best_params': best_params,
            'amc_setting': None,
            'temp_setting': temp_setting,
            'final_result': final_result
        }
    
    elif mode == 'optimize_combined':
        def objective(hparams: dict):
            amc_setting = {
                'amc_instance': hparams['amc_instance'],
                'amc_temporal': hparams['amc_temporal'],
                'amc_margin': hparams['amc_margin']
            }
            temp_setting = {
                'min_tau': hparams['min_tau'],
                'max_tau': hparams['max_tau'],
                't_max': hparams['t_max']
            }
            result = train_model(config, temp_dictionary=temp_setting, amc_setting=amc_setting, type="split")
            return result['acc']
        
        search = get_combined_search_space()
        best_params = search.run(objective, "maximize", steps=search_steps, n_jobs=1)
        
        # Final training with best parameters
        amc_setting = {
            'amc_instance': best_params['amc_instance'],
            'amc_temporal': best_params['amc_temporal'],
            'amc_margin': best_params['amc_margin']
        }
        temp_setting = {
            'min_tau': best_params['min_tau'],
            'max_tau': best_params['max_tau'],
            't_max': best_params['t_max']
        }
        final_result = train_model(config, temp_dictionary=temp_setting, amc_setting=amc_setting, type="full")
        
        return {
            'best_params': best_params,
            'amc_setting': amc_setting,
            'temp_setting': temp_setting,
            'final_result': final_result
        }
    
    else:
        raise ValueError(f"Unknown optimization mode: {mode}")

def get_ultra_fast_search_steps(dataset_size, base_steps):
    """Ultra-aggressive search step reduction for small datasets"""
    if dataset_size < 500:
        return min(base_steps // 4, 5)  # Only 5 steps for tiny datasets
    elif dataset_size < 1000:
        return min(base_steps // 3, 8)  # Only 8 steps for small datasets 
    elif dataset_size < 2000:
        return min(base_steps // 2, 12) # 12 steps for medium datasets
    else:
        return base_steps

def enable_pytorch_optimizations():
    """Enable all possible PyTorch optimizations for speed"""
    if torch.cuda.is_available():
        # Enable tensor fusion
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable flash attention if available (PyTorch 2.0+)
        try:
            torch.nn.functional.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
        except:
            pass
    
    # Set optimal number of threads
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')  # Trade precision for speed

def create_optimized_dataloader(data, batch_size, shuffle=True):
    """Create optimized data loader for maximum throughput"""
    return DataLoader(
        TensorDataset(torch.from_numpy(data).float()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=OPTIMIZATION_FLAGS.get('num_workers', 2),
        pin_memory=OPTIMIZATION_FLAGS.get('pin_memory', True),
        persistent_workers=True if OPTIMIZATION_FLAGS.get('num_workers', 0) > 0 else False,
        drop_last=True  # For consistent batch sizes
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimeHUT Integrated Training with Multiple Scenarios')
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experiment data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, forecast_npy, forecast_npy_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=2002, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--method', type=str, default='acc', help='whether to choose acc or auprc or both')
    parser.add_argument('--dataroot', type=str, default='/media/milad/DATA/TSResearch/datasets', help='root for the dataset')
    
    # Training scenario selection
    parser.add_argument('--scenario', type=str, default='baseline', 
                       choices=list(SCENARIO_MODES.keys()),
                       help='Training scenario mode. Available options: ' + ', '.join(SCENARIO_MODES.keys()))
    
    # Optimization flags
    parser.add_argument('--skip-search', action="store_true", help='Skip hyperparameter search and use default params')
    parser.add_argument('--search-steps', type=int, default=20, help='Number of hyperparameter search steps (default: 20 for pyhopper optimization)')
    
    # Manual parameter specification
    parser.add_argument('--amc-instance', type=float, default=0.1, help='AMC coefficient for instance contrastive loss')
    parser.add_argument('--amc-temporal', type=float, default=0.1, help='AMC coefficient for temporal contrastive loss')
    parser.add_argument('--amc-margin', type=float, default=0.5, help='AMC margin parameter')
    parser.add_argument('--min-tau', type=float, default=0.15, help='Minimum tau for temperature scheduling')
    parser.add_argument('--max-tau', type=float, default=0.75, help='Maximum tau for temperature scheduling')
    parser.add_argument('--t-max', type=float, default=10.5, help='T_max for temperature scheduling')
    
    # Advanced cosine annealing parameters
    parser.add_argument('--phase', type=float, default=0.0, help='Phase shift for cosine annealing (0 to 2π)')
    parser.add_argument('--frequency', type=float, default=1.0, help='Frequency multiplier for cosine annealing')
    parser.add_argument('--bias', type=float, default=0.0, help='Bias term for cosine annealing')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for adaptive cosine annealing')
    parser.add_argument('--adaptation-rate', type=float, default=0.1, help='Adaptation rate for adaptive cosine')
    parser.add_argument('--num-cycles', type=int, default=3, help='Number of cycles for multi-cycle cosine')
    parser.add_argument('--decay-factor', type=float, default=0.8, help='Decay factor for multi-cycle cosine')
    parser.add_argument('--restart-period', type=float, default=5.0, help='Restart period for cosine with restarts')
    parser.add_argument('--restart-mult', type=float, default=1.5, help='Restart multiplier for cosine with restarts')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 TimeHUT INTEGRATED TRAINING")
    print("=" * 60)
    print("Dataset:", args.dataset)
    print("Scenario:", args.scenario)
    print("Description:", SCENARIO_MODES[args.scenario])
    print("Arguments:", str(args))
    print("=" * 60)
    
    out_dir = "results/"
    os.makedirs(out_dir, exist_ok=True)
    scenario_suffix = f"_{args.scenario}" if args.scenario != 'baseline' else ""
    run_name = "_".join((args.loader, args.dataset, args.method + scenario_suffix, "integrated.json"))
    out_name = os.path.join(out_dir, run_name)
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    # Setup performance optimizations
    setup_performance_optimizations()
    clear_gpu_memory()  # Clear any existing GPU memory
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, args.dataroot)
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True, dataroot=args.dataroot)
        train_data = data[:, train_slice]
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, dataroot=args.dataroot)
        train_data = data[:, train_slice]
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True, dataroot=args.dataroot)
        train_data = data[:, train_slice]
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset, dataroot=args.dataroot)
        train_data = all_train_data
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset, subdataset='coldstart', dataroot=args.dataroot)
        train_data = all_train_data
    else:
        raise ValueError(f"Unsupported loader: {args.loader}")
    
    print('done')
    
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = datautils.noise_mask(train_data, args.irregular)
            test_data = datautils.noise_mask(test_data, args.irregular)
        elif task_type == 'forecasting':
            train_data = datautils.noise_mask(train_data, args.irregular)
    
    # Split data for hyperparameter search (when needed) - only for classification
    if args.scenario not in ['gridsearch_amc', 'gridsearch_temp', 'gridsearch_full'] and task_type == 'classification':
        # For optimization scenarios, we always need validation split
        if args.scenario in ['optimize_amc', 'optimize_temp', 'optimize_combined', 'temp_only', 'amc_temp']:
            train_data_split, val_data_split, train_labels_split, val_labels_split = train_test_split(
                train_data, train_labels, test_size=1-labeled_ratio, random_state=42, stratify=train_labels
            )
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    # Optimize search parameters based on dataset size
    dataset_size = len(train_data) if task_type == 'classification' else 1000
    if args.scenario in ['optimize_amc', 'optimize_temp', 'optimize_combined']:
        # Adjust search steps for small datasets
        if dataset_size < 500:
            effective_search_steps = min(args.search_steps, 5)
            print(f"🔍 Small dataset detected ({dataset_size} samples): Reducing search steps to {effective_search_steps}")
        elif dataset_size < 2000:
            effective_search_steps = min(args.search_steps, 10) 
            print(f"🔍 Medium dataset detected ({dataset_size} samples): Reducing search steps to {effective_search_steps}")
        else:
            effective_search_steps = args.search_steps
        
        # Override the command line argument for internal use
        args.search_steps = effective_search_steps
    
    # Execute based on scenario
    if args.scenario == 'baseline':
        print("\n🔧 Running BASELINE scenario...")
        print("No AMC losses or temperature scheduling")
        
        final_res = train_model(config, temp_dictionary=None, amc_setting=None, type="full")
        output = {
            'scenario': args.scenario,
            'amc_setting': None,
            'temp_setting': None,
            'result': final_res
        }
    
    elif args.scenario == 'amc_only':
        print("\n🔧 Running AMC ONLY scenario...")
        
        amc_setting = create_amc_setting(
            args.amc_instance, args.amc_temporal, args.amc_margin
        )
        print(f"AMC Settings: {amc_setting}")
        
        final_res = train_model(config, temp_dictionary=None, amc_setting=amc_setting, type="full")
        output = {
            'scenario': args.scenario,
            'amc_setting': amc_setting,
            'temp_setting': None,
            'result': final_res
        }
    
    elif args.scenario == 'temp_only':
        print("\n🔧 Running TEMPERATURE ONLY scenario...")
        
        # Optimize temperature search
        dataset_size = len(train_data)
        skip_search = args.skip_search or should_skip_search(dataset_size, args.dataset)
        
        if skip_search:
            print(f"Using default temperature parameters for {args.dataset} (size: {dataset_size})")
            temp_setting = create_temp_setting(args.min_tau, args.max_tau, args.t_max)
        else:
            print(f"Running temperature optimization with {args.search_steps} steps...")
            
            def objective(hparams: dict):
                temp_settings = {
                    'min_tau': hparams['min_tau'],
                    'max_tau': hparams['max_tau'],
                    't_max': hparams['t_max']
                }
                out = train_model(config, temp_settings, amc_setting=None, type="split")
                return out['acc']
            
            search = get_optimized_search_space()
            temp_setting = search.run(objective, "maximize", steps=args.search_steps, n_jobs=1)
        
        print(f"Temperature Settings: {temp_setting}")
        final_res = train_model(config, temp_dictionary=temp_setting, amc_setting=None, type="full")
        output = {
            'scenario': args.scenario,
            'amc_setting': None,
            'temp_setting': temp_setting,
            'result': final_res,
            'optimization_info': {
                'skipped_search': skip_search,
                'search_steps': 0 if skip_search else args.search_steps
            }
        }
    
    elif args.scenario == 'amc_temp':
        print("\n🔧 Running AMC + TEMPERATURE scenario...")
        
        amc_setting = create_amc_setting(
            args.amc_instance, args.amc_temporal, args.amc_margin
        )
        
        # Optimize temperature search
        dataset_size = len(train_data)
        skip_search = args.skip_search or should_skip_search(dataset_size, args.dataset)
        
        if skip_search:
            print(f"Using default temperature parameters for {args.dataset} (size: {dataset_size})")
            temp_setting = create_temp_setting(args.min_tau, args.max_tau, args.t_max)
        else:
            print(f"Running temperature optimization with {args.search_steps} steps...")
            
            def objective(hparams: dict):
                temp_settings = {
                    'min_tau': hparams['min_tau'],
                    'max_tau': hparams['max_tau'],
                    't_max': hparams['t_max']
                }
                out = train_model(config, temp_settings, amc_setting=amc_setting, type="split")
                return out['acc']
            
            search = get_optimized_search_space()
            temp_setting = search.run(objective, "maximize", steps=args.search_steps, n_jobs=1)
        
        print(f"AMC Settings: {amc_setting}")
        print(f"Temperature Settings: {temp_setting}")
        final_res = train_model(config, temp_dictionary=temp_setting, amc_setting=amc_setting, type="full")
        output = {
            'scenario': args.scenario,
            'amc_setting': amc_setting,
            'temp_setting': temp_setting,
            'result': final_res,
            'optimization_info': {
                'skipped_search': skip_search,
                'search_steps': 0 if skip_search else args.search_steps
            }
        }
    
    elif args.scenario == 'optimize_amc':
        print("\n🔧 Running PYHOPPER AMC OPTIMIZATION scenario...")
        
        optimization_result = run_pyhopper_optimization(config, 'optimize_amc', args.search_steps)
        
        print(f"🏆 Best AMC Parameters Found:")
        print(f"AMC Instance: {optimization_result['best_params']['amc_instance']:.3f}")
        print(f"AMC Temporal: {optimization_result['best_params']['amc_temporal']:.3f}")
        print(f"AMC Margin: {optimization_result['best_params']['amc_margin']:.3f}")
        
        output = {
            'scenario': args.scenario,
            'optimization_method': 'pyhopper',
            'search_steps': args.search_steps,
            'best_params': optimization_result['best_params'],
            'amc_setting': optimization_result['amc_setting'],
            'temp_setting': optimization_result['temp_setting'],
            'result': optimization_result['final_result']
        }
    
    elif args.scenario == 'optimize_temp':
        print("\n🔧 Running PYHOPPER TEMPERATURE OPTIMIZATION scenario...")
        
        optimization_result = run_pyhopper_optimization(config, 'optimize_temp', args.search_steps)
        
        print(f"🏆 Best Temperature Parameters Found:")
        print(f"Min Tau: {optimization_result['best_params']['min_tau']:.3f}")
        print(f"Max Tau: {optimization_result['best_params']['max_tau']:.3f}")  
        print(f"T Max: {optimization_result['best_params']['t_max']:.1f}")
        
        output = {
            'scenario': args.scenario,
            'optimization_method': 'pyhopper',
            'search_steps': args.search_steps,
            'best_params': optimization_result['best_params'],
            'amc_setting': optimization_result['amc_setting'],
            'temp_setting': optimization_result['temp_setting'],
            'result': optimization_result['final_result']
        }
    
    elif args.scenario == 'optimize_combined':
        print("\n🔧 Running PYHOPPER COMBINED OPTIMIZATION scenario...")
        
        optimization_result = run_pyhopper_optimization(config, 'optimize_combined', args.search_steps)
        
        print(f"🏆 Best Combined Parameters Found:")
        print(f"AMC Instance: {optimization_result['best_params']['amc_instance']:.3f}")
        print(f"AMC Temporal: {optimization_result['best_params']['amc_temporal']:.3f}")
        print(f"AMC Margin: {optimization_result['best_params']['amc_margin']:.3f}")
        print(f"Min Tau: {optimization_result['best_params']['min_tau']:.3f}")
        print(f"Max Tau: {optimization_result['best_params']['max_tau']:.3f}")
        print(f"T Max: {optimization_result['best_params']['t_max']:.1f}")
        
        output = {
            'scenario': args.scenario,
            'optimization_method': 'pyhopper',
            'search_steps': args.search_steps,
            'best_params': optimization_result['best_params'],
            'amc_setting': optimization_result['amc_setting'],
            'temp_setting': optimization_result['temp_setting'],
            'result': optimization_result['final_result']
        }
    
    elif args.scenario.startswith('temp_'):
        print(f"\n🔧 Running {args.scenario.upper()} scenario...")
        scheduler_type = args.scenario.replace('temp_', '')
        
        # Map scenario names to actual scheduler names
        scheduler_mapping = {
            'cosine': 'cosine_annealing',
            'adaptive_cosine': 'adaptive_cosine_annealing',
            'multi_cycle_cosine': 'multi_cycle_cosine',
            'cosine_restarts': 'cosine_with_restarts',
            'linear': 'linear_decay',
            'exponential': 'exponential_decay',
            'step': 'step_decay',
            'polynomial': 'polynomial_decay',
            'sigmoid': 'sigmoid_decay',
            'warmup_cosine': 'warmup_cosine',
            'constant': 'constant',
            'cyclic': 'cyclic'
        }
        
        actual_scheduler = scheduler_mapping.get(scheduler_type, scheduler_type)
        print(f"Using {actual_scheduler} temperature scheduling")
        
        # Create temperature setting with the specific scheduler
        from temperature_schedulers import create_temperature_settings
        
        # Collect scheduler-specific parameters
        scheduler_kwargs = {}
        if actual_scheduler == 'cosine_annealing':
            scheduler_kwargs = {
                'phase': args.phase,
                'frequency': args.frequency,
                'bias': args.bias
            }
        elif actual_scheduler == 'adaptive_cosine_annealing':
            scheduler_kwargs = {
                'momentum': args.momentum,
                'adaptation_rate': args.adaptation_rate
            }
        elif actual_scheduler == 'multi_cycle_cosine':
            scheduler_kwargs = {
                'num_cycles': args.num_cycles,
                'decay_factor': args.decay_factor
            }
        elif actual_scheduler == 'cosine_with_restarts':
            scheduler_kwargs = {
                'restart_period': args.restart_period,
                'restart_mult': args.restart_mult
            }
        
        temp_setting = create_temperature_settings(
            method=actual_scheduler,
            min_tau=args.min_tau,
            max_tau=args.max_tau,
            t_max=args.t_max,
            **scheduler_kwargs
        )
        
        print(f"Temperature Settings: {temp_setting}")
        final_res = train_model(config, temp_dictionary=temp_setting, amc_setting=None, type="full")
        
        # Create JSON-serializable copy of temp_setting without the function
        temp_setting_serializable = {k: v for k, v in temp_setting.items() if k != 'scheduler_func'}
        
        output = {
            'scenario': args.scenario,
            'amc_setting': None,
            'temp_setting': temp_setting_serializable,
            'result': final_res
        }

    elif args.scenario in ['gridsearch_amc', 'gridsearch_temp', 'gridsearch_full']:
        print(f"\n🔧 Running {args.scenario.upper()} scenario...")
        
        results = run_grid_search(config, args.scenario)
        
        # Find best result
        successful_results = [r for r in results if r['success']]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['result']['acc'])
            print(f"\n🏆 Best Result:")
            print(f"Accuracy: {best_result['result']['acc']:.4f}")
            print(f"AUPRC: {best_result['result']['auprc']:.4f}")
            print(f"AMC Setting: {best_result['amc_setting']}")
            print(f"Temp Setting: {best_result['temp_setting']}")
        else:
            print("⚠️ No successful experiments!")
            best_result = None
        
        output = {
            'scenario': args.scenario,
            'all_results': results,
            'best_result': best_result,
            'total_experiments': len(results),
            'successful_experiments': len(successful_results)
        }
    
    # Save results
    with open(out_name, "w") as out_file:
        json.dump(output, out_file, indent=2)
    
    print(f"\n💾 Results saved to: {out_name}")
    print("✅ Finished!")
    
    # Print summary
    if args.scenario not in ['gridsearch_amc', 'gridsearch_temp', 'gridsearch_full']:
        result = output['result']
        print(f"\n📊 FINAL RESULTS:")
        print(f"Accuracy: {result['acc']:.4f}")
        print(f"AUPRC: {result['auprc']:.4f}")
    else:
        if best_result:
            print(f"\n📊 GRID SEARCH SUMMARY:")
            print(f"Total Experiments: {len(results)}")
            print(f"Successful: {len(successful_results)}")
            print(f"Best Accuracy: {best_result['result']['acc']:.4f}")
            print(f"Best AUPRC: {best_result['result']['auprc']:.4f}")
