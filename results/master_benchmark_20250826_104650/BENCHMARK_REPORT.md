# TSlib Master Benchmark Report

**Generated:** 2025-08-26T10:46:58.362683
**Total Time:** 0.1 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.4667
- **Best Accuracy:** 0.4667

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| Ti_MAE | 0.4667 | 0.4667 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| AtrialFibrillation | 0.4667 | 0.4667 | Ti_MAE | 1 |

## Configuration

```json
{
  "models": [
    "Ti_MAE"
  ],
  "datasets": [
    "AtrialFibrillation"
  ],
  "use_original_iterations": true,
  "epochs": null,
  "batch_size": 8,
  "enable_optimization": true,
  "optimization_mode": "fair",
  "timeout_minutes": 200,
  "enable_gpu_monitoring": true,
  "save_intermediate_results": false,
  "fair_comparison_mode": true
}
```
