# TSlib Master Benchmark Report

**Generated:** 2025-08-26T10:49:08.347926
**Total Time:** 0.1 minutes

## Summary Statistics

- **Total Experiments:** 1
- **Successful Runs:** 1
- **Success Rate:** 100.00%
- **Average Accuracy:** 0.3333
- **Best Accuracy:** 0.3333

## Model Performance

| Model | Avg Accuracy | Best Accuracy | Datasets Tested |
|-------|-------------|---------------|----------------|
| VQ_MTM | 0.3333 | 0.3333 | 1 |

## Dataset Performance

| Dataset | Avg Accuracy | Best Accuracy | Best Model | Models Tested |
|---------|-------------|---------------|------------|---------------|
| AtrialFibrillation | 0.3333 | 0.3333 | VQ_MTM | 1 |

## Configuration

```json
{
  "models": [
    "VQ_MTM"
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
