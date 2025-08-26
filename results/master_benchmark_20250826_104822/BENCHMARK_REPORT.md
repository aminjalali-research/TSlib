# TSlib Master Benchmark Report

**Generated:** 2025-08-26T10:48:39.378636
**Total Time:** 0.3 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.2667
- **Best Accuracy:** 0.2667

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| TFC | 0.2667 | 0.2667 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| AtrialFibrillation | 0.2667 | 0.2667 | TFC | 1 |

## Configuration

```json
{
  "models": [
    "TFC"
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
