# Enhanced Metrics Collection System
## Time/Epoch, Peak GPU Memory, and FLOPs/Epoch

This standalone system provides comprehensive enhanced metrics collection for TSlib models without interfering with existing files.

## 🌟 Enhanced Metrics Collected

✅ **Time/Epoch** - Average training time per epoch  
✅ **Peak GPU Memory** - Maximum GPU memory usage during training  
✅ **FLOPs/Epoch** - Floating point operations per training epoch  
✅ **Real-time GPU Monitoring** - Continuous resource tracking  
✅ **Computational Efficiency** - FLOPs efficiency, Memory efficiency  
✅ **Training Dynamics** - Epoch progression analysis  

## 📁 System Structure

```
enhanced_metrics/
├── enhanced_single_model_runner.py  # Single model runner with enhanced metrics
├── enhanced_batch_runner.py         # Batch runner for multiple experiments  
├── example_config.json              # Configuration file example
├── README.md                        # This documentation
├── results/                         # Individual model results
└── batch_results/                   # Batch experiment results
```

## 🚀 Usage Examples

### Single Model Testing

```bash
# Basic usage
python enhanced_metrics/enhanced_single_model_runner.py TimesURL Chinatown

# With custom timeout (180 seconds)
python enhanced_metrics/enhanced_single_model_runner.py BIOT AtrialFibrillation 180

# Test computational efficiency
python enhanced_metrics/enhanced_single_model_runner.py CoST Chinatown 300
```

### Batch Testing

```bash
# Using configuration file
python enhanced_metrics/enhanced_batch_runner.py --config enhanced_metrics/example_config.json

# Command-line specification
python enhanced_metrics/enhanced_batch_runner.py --models TimesURL,BIOT,CoST --datasets Chinatown,AtrialFibrillation

# With custom timeout
python enhanced_metrics/enhanced_batch_runner.py --config enhanced_metrics/example_config.json --timeout 180
```

## 📊 Output Examples

### Single Model Output
```
⭐ ENHANCED METRICS:
   📅 Time/Epoch: 12.34s
   🔥 Peak GPU Memory: 2048MB (2.00GB)
   ⚡ FLOPs/Epoch: 2.45e+09

🚀 EFFICIENCY ANALYSIS:
   Accuracy/Second: 0.000123
   FLOPs Efficiency: 0.000456 accuracy/GFLOP
   Memory Efficiency: 0.4567 accuracy/GB
   Time Efficiency: 0.000789 accuracy/s per epoch
```

### Batch Summary Output
```
🏆 PERFORMANCE CHAMPIONS:
   🎯 Accuracy: TimesURL on Chinatown = 0.9854
   ⚡ FLOPs Efficiency: CoST on AtrialFibrillation = 0.000234 acc/GFLOP
   💾 Memory Efficiency: BIOT on Chinatown = 0.1234 acc/GB
   ⏱️ Time Efficiency: TFC on AtrialFibrillation = 0.000456 acc/s per epoch
```

## 📋 Configuration File Format

```json
{
  "models": [
    "TimesURL",
    "BIOT", 
    "CoST",
    "TFC",
    "Ti_MAE",
    "TS2vec",
    "TimeHUT",
    "SoftCLT"
  ],
  "datasets": [
    "Chinatown",
    "AtrialFibrillation",
    "CricketX",
    "MotorImagery"
  ]
}
```

## 🔧 Technical Details

### Enhanced Metrics Definitions

- **Time/Epoch**: Average training time per epoch (seconds)
- **Peak GPU Memory**: Maximum GPU memory usage during training (MB/GB)
- **FLOPs/Epoch**: Floating point operations per training epoch
- **FLOPs Efficiency**: Accuracy per GFLOP (higher is better)
- **Memory Efficiency**: Accuracy per GB of peak memory (higher is better)
- **Time Efficiency**: Accuracy per second per epoch (higher is better)

### Dependencies

Required packages (graceful fallback if missing):
- `GPUtil` for GPU monitoring
- `psutil` for system monitoring
- `numpy` for statistical analysis
- `torch` for CUDA information (optional)

### Real-time Monitoring

The system includes background GPU monitoring that:
- Samples GPU usage every 500ms
- Tracks peak memory, temperature, utilization
- Provides comprehensive resource analysis
- Handles monitoring failures gracefully

## 📂 Result Files

### Single Model Results
- `{model}_{dataset}_enhanced_metrics_{timestamp}.json` - Detailed results
- Contains all enhanced metrics, efficiency analysis, resource usage

### Batch Results  
- `enhanced_batch_results_{timestamp}.json` - Comprehensive batch analysis
- `enhanced_batch_summary_{timestamp}.csv` - CSV for easy analysis
- `performance_champions_{timestamp}.json` - Champions across metrics

## 🎯 Model Families Supported

- **TS2vec Family**: TS2vec, TimeHUT, SoftCLT, TimesURL
- **VQ-MTM Family**: BIOT, VQ_MTM, Ti_MAE, SimMTM, TimesNet, DCRNN
- **MF-CLR Family**: TFC, CoST, CPC, TNC, TS_TCC, TLoss, MF_CLR

## 📊 Supported Datasets

- **UCR**: Chinatown, CricketX, EigenWorms, EOGVerticalSignal
- **UEA**: AtrialFibrillation, MotorImagery, StandWalkJump, GesturePebbleZ1

## 🛠️ Advanced Usage

### Custom FLOPs Estimation

The system includes intelligent FLOPs estimation based on:
- Model architectural complexity
- Dataset characteristics (timesteps, channels, samples)
- Batch size and epoch configuration

### Efficiency Classifications

Models are automatically classified into:
- **Runtime**: very_fast, fast, medium, slow, very_slow
- **Performance**: excellent, good, moderate, poor, very_poor  
- **Memory**: low, medium, high, very_high
- **Computational Intensity**: high, medium, low, very_low

## 🔍 Troubleshooting

### GPU Monitoring Issues
```bash
# Install GPUtil if not available
pip install GPUtil

# Check GPU availability
python -c "import GPUtil; print(GPUtil.getGPUs())"
```

### Environment Issues
- Uses existing TSlib conda environments
- Automatically handles MKL threading conflicts
- Graceful fallback for missing dependencies

## 📈 Performance Analysis

The enhanced metrics system provides:
- Statistical analysis across batch runs
- Model family performance comparison
- Resource utilization trends
- Computational efficiency rankings
- Performance champion identification

## 🔐 Safety Features

- **Standalone Architecture**: No interference with existing TSlib files
- **Separate Results Directory**: All outputs isolated
- **Graceful Error Handling**: Continues batch runs even with individual failures
- **Resource Monitoring**: Real-time tracking prevents resource exhaustion
- **Timeout Protection**: Configurable timeouts prevent hanging

---

## Quick Start

1. **Test single model**: `python enhanced_metrics/enhanced_single_model_runner.py TimesURL Chinatown`
2. **Run batch**: `python enhanced_metrics/enhanced_batch_runner.py --config enhanced_metrics/example_config.json`
3. **View results**: Check `enhanced_metrics/results/` and `enhanced_metrics/batch_results/`

This system provides comprehensive enhanced metrics collection that complements the existing TSlib benchmarking infrastructure while maintaining complete independence.
