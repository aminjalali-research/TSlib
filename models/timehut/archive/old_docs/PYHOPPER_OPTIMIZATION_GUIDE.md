# TimeHUT Pyhopper Optimization Guide

## 🎯 Overview

The TimeHUT training system now includes advanced hyperparameter optimization using **pyhopper**, allowing for intelligent search over AMC and temperature parameters. This provides more efficient optimization compared to grid search while finding better parameter combinations.

## 🚀 New Optimization Scenarios

### 1. **optimize_amc** - AMC Parameter Optimization
Optimizes AMC (Adaptive Margin Contrastive) parameters using pyhopper:
- `amc_instance`: Instance-wise contrastive loss coefficient (0.0 - 5.0)
- `amc_temporal`: Temporal contrastive loss coefficient (0.0 - 5.0)  
- `amc_margin`: Margin parameter for contrastive loss (0.1 - 1.0)

```bash
# Basic AMC optimization
python train_optimized.py SyntheticControl amc_exp --loader UCR --scenario optimize_amc --epochs 100

# With custom search steps
python train_optimized.py SyntheticControl amc_exp --loader UCR --scenario optimize_amc --epochs 100 --search-steps 30
```

### 2. **optimize_temp** - Temperature Scheduling Optimization
Optimizes temperature scheduling parameters:
- `min_tau`: Minimum temperature value (0.07 - 0.3)
- `max_tau`: Maximum temperature value (0.6 - 1.0)
- `t_max`: Temperature schedule duration (5 - 20)

```bash
# Basic temperature optimization
python train_optimized.py SyntheticControl temp_exp --loader UCR --scenario optimize_temp --epochs 100

# With more search steps for better results
python train_optimized.py SyntheticControl temp_exp --loader UCR --scenario optimize_temp --epochs 100 --search-steps 25
```

### 3. **optimize_combined** - Joint AMC + Temperature Optimization
Optimizes both AMC and temperature parameters simultaneously for best combined performance:

```bash
# Joint optimization (recommended for best results)
python train_optimized.py SyntheticControl combined_exp --loader UCR --scenario optimize_combined --epochs 100 --search-steps 40

# Quick combined optimization
python train_optimized.py SyntheticControl combined_exp --loader UCR --scenario optimize_combined --epochs 100 --search-steps 20
```

## ⚙️ Optimization Parameters

### Search Steps Recommendations
- **Quick optimization**: `--search-steps 10-15`
- **Standard optimization**: `--search-steps 20-30` (default: 20)
- **Thorough optimization**: `--search-steps 40-60`
- **Extensive optimization**: `--search-steps 100+`

### Parameter Ranges (Pyhopper Search Space)

#### AMC Parameters
```python
amc_instance: 0.0 to 5.0 (step: 0.1)
amc_temporal: 0.0 to 5.0 (step: 0.1)  
amc_margin: 0.1 to 1.0 (step: 0.1)
```

#### Temperature Parameters
```python
min_tau: 0.07 to 0.3 (step: 0.01)
max_tau: 0.6 to 1.0 (step: 0.02)
t_max: 5.0 to 20.0 (step: 1.0)
```

## 📊 Performance Comparison

### Optimization Methods Comparison
| Method | Speed | Coverage | Quality | Use Case |
|--------|--------|----------|---------|----------|
| **Grid Search** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Comprehensive testing |
| **Pyhopper Optimization** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Best performance |
| **Manual Tuning** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | Quick experiments |

### Expected Search Times
- **AMC Only**: ~15-30 minutes (20 steps)
- **Temperature Only**: ~20-40 minutes (20 steps)  
- **Combined**: ~30-60 minutes (40 steps)

*Times depend on dataset size, model complexity, and hardware*

## 🎯 Best Practices

### 1. **Start with Combined Optimization**
```bash
# Recommended first approach
python train_optimized.py MyDataset exp1 --loader UCR --scenario optimize_combined --epochs 100 --search-steps 30
```

### 2. **Use Appropriate Search Steps**
- Small datasets (< 1000 samples): 15-20 steps
- Medium datasets (1000-10000): 20-30 steps  
- Large datasets (> 10000): 30-50 steps

### 3. **Progressive Optimization**
```bash
# Step 1: Quick combined optimization
python train_optimized.py MyDataset exp1 --loader UCR --scenario optimize_combined --search-steps 15

# Step 2: Refine best approach (if AMC was dominant)
python train_optimized.py MyDataset exp2 --loader UCR --scenario optimize_amc --search-steps 25

# Step 3: Final validation
python train_optimized.py MyDataset exp3 --loader UCR --scenario optimize_combined --search-steps 40
```

## 📈 Example Workflows

### For Research Paper Results
```bash
# Comprehensive evaluation across multiple datasets
for dataset in SyntheticControl TwoLeadECG ECG200 Coffee; do
    echo "Processing $dataset..."
    python train_optimized.py $dataset ${dataset}_opt --loader UCR \
        --scenario optimize_combined --epochs 100 --search-steps 50
done
```

### For Quick Experiments
```bash
# Fast parameter exploration
python train_optimized.py MyDataset quick_test --loader UCR \
    --scenario optimize_combined --epochs 50 --search-steps 10
```

### For Production Deployment
```bash
# Thorough optimization for deployment
python train_optimized.py ProductionData final_model --loader UCR \
    --scenario optimize_combined --epochs 200 --search-steps 100
```

## 🔍 Understanding Results

### Output Structure
```json
{
    "scenario": "optimize_combined",
    "optimization_method": "pyhopper", 
    "search_steps": 30,
    "best_params": {
        "amc_instance": 1.234,
        "amc_temporal": 0.567,
        "amc_margin": 0.789,
        "min_tau": 0.123,
        "max_tau": 0.845,
        "t_max": 12.3
    },
    "amc_setting": {...},
    "temp_setting": {...},
    "result": {
        "acc": 0.9234,
        "auprc": 0.8567
    }
}
```

### Key Metrics to Monitor
1. **Final Accuracy/AUPRC**: Primary performance metrics
2. **Parameter Values**: Optimal hyperparameter settings
3. **Search Convergence**: How quickly optimization converged
4. **Training Time**: Total time including optimization

## 🚫 Common Issues & Solutions

### Issue 1: Optimization Not Converging
**Solution**: Increase search steps or check data split quality
```bash
--search-steps 50  # Instead of 20
```

### Issue 2: AMC Parameters All Near Zero
**Solution**: Check if AMC losses are properly integrated
```bash
# Try AMC-only first to verify functionality
--scenario optimize_amc
```

### Issue 3: Temperature Parameters at Boundaries  
**Solution**: Expand search ranges or check temperature scheduling implementation
```bash
# Manual verification
--scenario temp_only --min-tau 0.05 --max-tau 1.2 --t-max 25
```

## 🔧 Advanced Usage

### Custom Parameter Ranges
You can modify the search spaces in `train_optimized.py`:

```python
def get_combined_search_space():
    return pyhopper.Search({
        # Expand AMC ranges for aggressive search
        "amc_instance": pyhopper.float(0.0, 10.0, "0.1f"),
        "amc_temporal": pyhopper.float(0.0, 10.0, "0.1f"),
        # Narrow temperature ranges based on prior knowledge  
        "min_tau": pyhopper.float(0.1, 0.2, "0.01f"),
        "max_tau": pyhopper.float(0.7, 0.9, "0.02f"),
    })
```

### Multi-Objective Optimization
For future extensions, consider optimizing both accuracy and AUPRC:

```python
def multi_objective(hparams):
    result = train_model(...)
    # Weighted combination
    return 0.7 * result['acc'] + 0.3 * result['auprc']
```

---

**🎯 Recommended Starting Point**: Use `--scenario optimize_combined --search-steps 30` for most datasets to get excellent results with reasonable computational cost.
