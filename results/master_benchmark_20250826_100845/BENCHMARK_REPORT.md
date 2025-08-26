# TSlib Master Benchmark Report

**Generated:** 2025-08-26T10:08:55.233374
**Total Time:** 0.2 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.5948
- **Best Accuracy:** 0.5948

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| TFC | 0.5948 | 0.5948 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| Chinatown | 0.5948 | 0.5948 | TFC | 1 |

## Configuration

```json
{
  "models": [
    "TFC"
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
