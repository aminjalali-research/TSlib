# TimeHUT Configuration Documentation

This document consolidates information from various TimeHUT training scripts and configurations that were present in the original repository.

## Overview

TimeHUT is a TS2Vec-based time series representation learning method that incorporates:
- Hyperparameter optimization using pyhopper
- Adaptive temperature scheduling
- Multiple training variants for different tasks

## Core Training Parameters

### Standard Configuration
```python
config = {
    'batch_size': 8,
    'lr': 0.001,
    'output_dims': 320,  # representation dimensions
    'max_train_length': 3000
}
```

### Temperature Settings (Hyperparameter Search)
```python
temp_settings = {
    'min_tau': 0.0-0.3,    # Range for minimum temperature
    'max_tau': 0.5-1.0,    # Range for maximum temperature  
    't_max': 1.0-20.0      # Range for temperature schedule max
}
```

### AMC (Adaptive Masking and Contrastive) Settings
```python
amc_setting = {
    'alpha': 0.1-1.0,      # AMC alpha parameter
    'beta': 0.1-1.0,       # AMC beta parameter
    'gamma': 0.1-1.0       # AMC gamma parameter
}
```

## Training Variants

### 1. Standard Training (`train.py`)
- Basic TimeHUT training with hyperparameter optimization
- Uses pyhopper for temperature parameter search
- Splits data 70/30 for validation
- Performs grid search over temperature parameters

### 2. Grid Search Training (`train_gridsearch.py`)
- More extensive hyperparameter search
- Commented out split validation (uses full training)
- Similar to standard but with different search strategies

### 3. AMC Training (`train_gridsearch_AMC.py`)
- Includes Adaptive Masking and Contrastive learning
- Additional AMC parameters in model initialization
- Supports classification, forecasting, and anomaly detection tasks

### 4. ETT Forecasting Variants
- `train_gridsearch_TimeCAST_ETT*.py`: Specialized for ETT datasets
- TimeCAST integration for forecasting tasks
- Various configurations for ETT1, ETT2, etc.

### 5. Temperature-focused Variants
- `train_gridsearch_temp.py`: Focus on temperature parameter optimization
- `train_gridsearch_AMC_temp.py`: Combines AMC with temperature optimization

## Dataset Support

### UCR Datasets
Standard parameters used:
```bash
--batch-size 8
--repr-dims 320
--max-threads 8
--seed 42
--eval
```

### UEA Datasets  
Similar configuration with loader flag:
```bash
--loader UEA
```

### Forecasting Datasets
- ETT (Electricity Transformer Temperature)
- Custom CSV formats
- Univariate and multivariate support

## Evaluation Protocols

### Classification
- SVM-based evaluation
- Reports accuracy and AUPRC
- Split validation during hyperparameter search

### Forecasting
- MSE and MAE metrics
- Multiple prediction horizons
- Scaler-aware evaluation

### Anomaly Detection
- Point anomaly detection
- Cold-start scenarios
- Time-delay considerations

## Key Hyperparameter Ranges

Based on analysis of all training scripts:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| min_tau | 0.0-0.3 | 0.15 | Minimum temperature |
| max_tau | 0.5-1.0 | 0.75 | Maximum temperature |
| t_max | 1.0-20.0 | 10.5 | Temperature schedule parameter |
| batch_size | 8-64 | 8 | Training batch size |
| repr_dims | 64-512 | 320 | Representation dimensions |
| lr | 0.0001-0.01 | 0.001 | Learning rate |
| alpha (AMC) | 0.1-1.0 | 0.5 | AMC alpha parameter |
| beta (AMC) | 0.1-1.0 | 0.5 | AMC beta parameter |
| gamma (AMC) | 0.1-1.0 | 0.5 | AMC gamma parameter |

## Usage Patterns

### For Classification (Our Use Case)
```bash
python train.py <dataset> <run_name> \
    --loader UCR \
    --batch-size 8 \
    --repr-dims 320 \
    --epochs <num_epochs> \
    --eval \
    --dataroot <path_to_datasets>
```

### Advanced Usage with AMC
```bash  
python train_gridsearch_AMC.py <dataset> <run_name> \
    --loader UCR \
    --batch-size 8 \
    --repr-dims 320 \
    --epochs <num_epochs> \
    --eval \
    --amc-alpha 0.5 \
    --amc-beta 0.5 \
    --amc-gamma 0.5
```

## Notes

- TimeHUT uses pyhopper for efficient hyperparameter optimization
- The method automatically optimizes temperature scheduling parameters
- AUPRC is reported alongside accuracy for classification tasks
- The method supports both split validation and full training modes
- Default hyperparameter search uses 10 steps for efficiency

## Files Preserved in Clean Version

Essential files kept after cleanup:
- `train.py` - Main training script (used in benchmarking)
- `ts2vec.py` - Core model implementation
- `datautils.py` - Data loading utilities  
- `utils.py` - General utilities
- `tasks/` - Evaluation protocols
- `models/` - Model components
- This documentation file

All grid search variants, shell scripts, and experimental files have been consolidated into this documentation and removed to keep the repository clean.
