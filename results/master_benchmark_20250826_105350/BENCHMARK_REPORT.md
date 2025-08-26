# TSlib Master Benchmark Report

**Generated:** 2025-08-26T10:53:59.120596
**Total Time:** 0.1 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.2667
- **Best Accuracy:** 0.2667

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| TS_TCC | 0.2667 | 0.2667 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| AtrialFibrillation | 0.2667 | 0.2667 | TS_TCC | 1 |

## Configuration

```json
{
  "models": [
    "TS_TCC"
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
