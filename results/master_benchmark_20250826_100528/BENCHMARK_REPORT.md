# TSlib Master Benchmark Report

**Generated:** 2025-08-26T10:08:10.594788
**Total Time:** 2.7 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.7668
- **Best Accuracy:** 0.7668

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| TLoss | 0.7668 | 0.7668 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| Chinatown | 0.7668 | 0.7668 | TLoss | 1 |

## Configuration

```json
{
  "models": [
    "TLoss"
  ],
  "datasets": [
    "Chinatown"
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
