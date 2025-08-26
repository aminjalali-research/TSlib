# TSlib Master Benchmark Report

**Generated:** 2025-08-26T10:05:18.168128
**Total Time:** 0.2 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.9300
- **Best Accuracy:** 0.9300

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| CPC | 0.9300 | 0.9300 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| Chinatown | 0.9300 | 0.9300 | CPC | 1 |

## Configuration

```json
{
  "models": [
    "CPC"
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
