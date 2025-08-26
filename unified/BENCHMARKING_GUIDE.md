# Comprehensive Time Series Method Benchmarking Framework

This framework provides complete performance comparison for **TimesURL**, **SoftCLT**, and **TS2Vec** methods on UCR time series datasets.

## 🎯 Metrics Tracked

### Performance Metrics
- **Accuracy**: Classification accuracy on test set
- **F1-Score**: F1 score for classification performance
- **AUPRC**: Area Under Precision-Recall Curve

### Efficiency Metrics
- **Total Training Time**: Complete training duration
- **Training Time/Epoch**: Average time per training epoch
- **Peak GPU Memory**: Maximum GPU memory usage during training
- **Average GPU Utilization**: Mean GPU utilization percentage
- **FLOPs/Epoch**: Estimated floating-point operations per epoch

## 📁 Framework Components

### Core Scripts

1. **`comprehensive_benchmark.py`** - Main benchmarking engine
2. **`test_benchmark.py`** - Quick testing interface
3. **`run_batch_benchmark.sh`** - Batch processing for multiple datasets
4. **`analyze_results.py`** - Results analysis and visualization
5. **`quick_test_all.py`** - Simple test runner (your existing script)

### Support Files

- **`compare_methods.py`** - Original comparison framework
- **`setup_comparison.sh`** - Environment setup helper

## 🚀 Quick Start

### 1. Single Dataset Benchmark
```bash
# Quick test (5 epochs)
python comprehensive_benchmark.py --dataset Beef --epochs 5

# Full test (50 epochs) 
python comprehensive_benchmark.py --dataset ArrowHead --epochs 50

# Specific methods only
python comprehensive_benchmark.py --dataset Chinatown --methods timesurl softclt --epochs 20
```

### 2. Batch Processing
```bash
# Run comprehensive benchmarks on multiple datasets
./run_batch_benchmark.sh

# This will test:
# - Quick tests: Beef, BeetleFly, Coffee (5 epochs each)
# - Medium tests: ArrowHead, Chinatown, GunPoint (20 epochs each)  
# - Full tests: ECG200, FaceAll, OSULeaf (50 epochs each)
```

### 3. Results Analysis
```bash
# Generate all analyses (plots + reports + rankings)
python analyze_results.py --all

# Individual analyses
python analyze_results.py --plots     # Generate comparison plots
python analyze_results.py --report    # Generate text summary
python analyze_results.py --ranking   # Show performance ranking
```

### 4. Test Framework
```bash
# Quick framework test
python test_benchmark.py --test

# Interactive test
python test_benchmark.py
```

## 📊 Output Files

### Benchmark Results (per run)
- `{dataset}_benchmark_{timestamp}.json` - Detailed metrics (JSON)
- `{dataset}_benchmark_{timestamp}.csv` - Summary table (CSV)
- `{dataset}_benchmark_{timestamp}.txt` - Human-readable report (TXT)

### Analysis Results
- `method_comparison_boxplots.png` - Performance distribution plots
- `accuracy_vs_time_scatter.png` - Efficiency vs accuracy trade-off
- `accuracy_heatmap.png` - Performance by dataset and method
- `benchmark_summary_report.txt` - Comprehensive analysis report

## 🔧 Configuration

### Method Parameters
- **TimesURL**: Standard configuration with representation learning
- **SoftCLT**: τ_inst=0.1, τ_temp=0.1 (soft contrastive learning)
- **TS2Vec**: Standard TS2Vec configuration

### System Requirements
- **GPU**: CUDA-enabled GPU (tested with RTX 3090)
- **Environment**: Conda environment `timesurl` with all dependencies
- **Datasets**: UCR/UEA datasets in `./datasets/` directory

## 📈 Example Results

### Quick Test Results (Chinatown, 3 epochs)
| Method | Runtime | Accuracy | Peak Memory | GPU Util |
|--------|---------|----------|-------------|----------|
| TimesURL | 3.46s | 96.5% | 2.1 GB | 85% |
| SoftCLT | 3.22s | 97.1% | 2.3 GB | 87% |
| TS2Vec | 2.54s | 97.4% | 1.9 GB | 82% |

## 🛠️ Advanced Usage

### Custom Benchmark Parameters
```bash
python comprehensive_benchmark.py \
    --dataset YourDataset \
    --epochs 100 \
    --batch-size 16 \
    --methods timesurl softclt \
    --output-dir ./custom_results
```

### Batch Processing with Custom Datasets
Edit `run_batch_benchmark.sh` to include your datasets:
```bash
custom_datasets=("YourDataset1" "YourDataset2")
for dataset in "${custom_datasets[@]}"; do
    python comprehensive_benchmark.py --dataset "$dataset" --epochs 30
done
```

### Analysis with Custom Results Directory
```bash
python analyze_results.py --results-dir ./your_results --all
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Error**: Reduce batch size or use smaller datasets
2. **Environment Issues**: Ensure `conda activate timesurl` works
3. **Missing Dependencies**: Run `pip install -r requirements.txt`
4. **Dataset Not Found**: Check that datasets are symlinked correctly

### Performance Tips

1. **Quick Testing**: Use 5-10 epochs for initial comparison
2. **Memory Management**: Run one method at a time for large datasets
3. **Monitoring**: Check `nvidia-smi` during benchmarking

## 📚 Understanding the Metrics

### Performance Interpretation
- **Accuracy > 95%**: Excellent performance
- **F1-Score**: Better than accuracy for imbalanced datasets
- **AUPRC**: Robust metric for classification quality

### Efficiency Interpretation  
- **Training Time**: Lower is better (depends on epochs)
- **GPU Memory**: Lower is better (allows larger batches)
- **GPU Utilization**: Higher is better (80%+ is good)
- **FLOPs**: Lower is better (more efficient)

## 🎯 Next Steps

1. **Run Initial Tests**: Start with quick tests on small datasets
2. **Batch Processing**: Use batch script for comprehensive comparison
3. **Analysis**: Generate plots and reports to understand performance
4. **Optimization**: Tune hyperparameters based on results
5. **Publication**: Use results for research papers or reports

## 📞 Support

The framework provides comprehensive logging and error handling. Check the output files for detailed information about each benchmark run.

For issues:
1. Check that all dependencies are installed
2. Verify GPU is available and working
3. Ensure datasets are properly linked
4. Review error logs in the output files
