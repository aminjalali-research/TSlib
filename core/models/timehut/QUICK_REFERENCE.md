# TimeHUT - Quick Reference Cheat Sheet

## 🚀 One-Liners for Common Tasks

### Pyhopper Optimization (RECOMMENDED) ⭐
```bash
# Best approach - joint optimization
python train_optimized.py DATASET EXPNAME --loader UCR --scenario optimize_combined --epochs 100

# AMC-focused optimization
python train_optimized.py DATASET EXPNAME --loader UCR --scenario optimize_amc --epochs 100

# Temperature-focused optimization  
python train_optimized.py DATASET EXPNAME --loader UCR --scenario optimize_temp --epochs 100
```

### Quick Experiments
```bash
# Baseline (no enhancements)
python train_optimized.py DATASET EXPNAME --loader UCR --scenario baseline --epochs 100

# Manual AMC tuning
python train_optimized.py DATASET EXPNAME --loader UCR --scenario amc_only --amc-instance 1.0 --amc-temporal 0.5 --epochs 100

# Manual combined
python train_optimized.py DATASET EXPNAME --loader UCR --scenario amc_temp --amc-instance 1.0 --epochs 100
```

### Comprehensive Search
```bash
# Full grid search (slow but complete)
python train_optimized.py DATASET EXPNAME --loader UCR --scenario gridsearch_full --epochs 100

# AMC grid search only
python train_optimized.py DATASET EXPNAME --loader UCR --scenario gridsearch_amc --epochs 100
```

## 🎯 Parameter Quick Guide

### Search Steps (for optimization scenarios)
- Quick: `--search-steps 10`
- Standard: `--search-steps 20` (default)
- Thorough: `--search-steps 40`
- Extensive: `--search-steps 80`

### AMC Parameters (for manual tuning)
- Instance: `--amc-instance 0.1` to `5.0` (typical: 0.5-2.0)
- Temporal: `--amc-temporal 0.1` to `5.0` (typical: 0.5-2.0)  
- Margin: `--amc-margin 0.1` to `1.0` (default: 0.5)

### Temperature Parameters (for manual tuning)
- Min Tau: `--min-tau 0.07` to `0.3` (typical: 0.1-0.2)
- Max Tau: `--max-tau 0.6` to `1.0` (typical: 0.7-0.9)
- T Max: `--t-max 5` to `20` (typical: 8-15)

## 📊 Scenario Selection Guide

| Use Case | Scenario | Typical Time | Quality |
|----------|----------|--------------|---------|
| **Quick test** | `baseline` | 5-10 min | ⭐⭐ |
| **Best performance** | `optimize_combined` | 30-60 min | ⭐⭐⭐⭐⭐ |
| **AMC research** | `optimize_amc` | 15-30 min | ⭐⭐⭐⭐ |
| **Temp research** | `optimize_temp` | 20-40 min | ⭐⭐⭐⭐ |
| **Manual tuning** | `amc_temp` + params | 5-10 min | ⭐⭐⭐ |
| **Comprehensive** | `gridsearch_full` | 2-8 hours | ⭐⭐⭐⭐ |

## 🗂️ File Locations

- **Main Script**: `train_optimized.py`
- **Results**: `results/LOADER_DATASET_METHOD_SCENARIO_integrated.json`
- **Loss Functions**: `models/losses_integrated.py`

## ⚡ Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Increase `--batch-size` for faster training (default: 8)
3. **Search Steps**: Balance time vs. quality (20 steps is usually good)
4. **Early Stop**: For optimization, model trains with reduced epochs first
5. **Memory**: GPU memory is automatically cleared between experiments

## 🚨 Common Issues

**Issue**: Out of memory  
**Solution**: Reduce `--batch-size` or use `--skip-search`

**Issue**: Optimization not improving  
**Solution**: Increase `--search-steps` or check data quality

**Issue**: Parameters at boundaries  
**Solution**: Check if manual params are reasonable

---

## 🎯 **TLDR - Start Here**

```bash
# For best results on most datasets:
python train_optimized.py MyDataset experiment --loader UCR --scenario optimize_combined --epochs 100

# Replace "MyDataset" with your dataset name
# Replace "experiment" with your experiment name  
# Change --loader based on your data type (UCR, UEA, forecast_csv, etc.)
```

**This single command will:**
- ✅ Automatically optimize both AMC and temperature parameters
- ✅ Train the final model with best parameters
- ✅ Save comprehensive results to JSON
- ✅ Handle GPU memory efficiently
- ✅ Provide detailed progress output

---

**📖 Full Documentation**: See `INTEGRATION_FINAL_SUMMARY.md` and `PYHOPPER_OPTIMIZATION_GUIDE.md`
