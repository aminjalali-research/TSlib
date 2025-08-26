# TSlib Master Benchmark Report

**Generated:** 2025-08-26T01:03:52.235896
**Total Time:** 0.3 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.9475
- **Best Accuracy:** 0.9475

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| CoST | 0.9475 | 0.9475 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| Chinatown | 0.9475 | 0.9475 | CoST | 1 |

## Configuration

```json
{
  "models": [
    "CoST"
  ],
  "datasets": [
    "Chinatown"
  ],
  "use_original_iterations": true,
  "epochs": null,
  "batch_size": 8,
  "enable_optimization": true,
  "optimization_mode": "fair",
  "timeout_minutes": 60,
  "enable_gpu_monitoring": true,
  "save_intermediate_results": false,
  "fair_comparison_mode": true
}
```
