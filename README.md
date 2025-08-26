# TSlib:  Time Series Self-Supervised Learning Benchmark

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Enhanced Metrics](https://img.shields.io/badge/Enhanced%20Metrics-Ready-brightgreen.svg)](enhanced_metrics/)

**TSlib** is a unified comprehensive benchmarking framework for time series self-supervised learning models, featuring advanced computational efficiency analysis and real-time GPU monitoring.

##  **Key Features**

### Comprehensive Model Coverage (25+ Validated Models)
-  TS2vec, TimeHUT, SoftCLT, TimesURL
-  BIOT, VQ_MTM, Ti_MAE, SimMTM, DCRNN
-  TFC, CoST, CPC, TNC, TS_TCC, TLoss, MF_CLR  


### Enhanced Metrics System 
- ** Time/Epoch Analysis**: Average training time per epoch
- ** Peak GPU Memory Tracking**: Maximum GPU memory usage monitoring
- ** FLOPs/Epoch Calculation**: Floating point operations per training epoch
- ** Computational Efficiency**: FLOPs efficiency, Memory efficiency, Time efficiency
- ** Real-time GPU Monitoring**: Continuous resource tracking during training
- ** Performance Champions**: Automatic identification of best performers

### **🔬 Scientific Benchmarking**
- **Fair Comparison Mode**: Standardized hyperparameters across all models
- **Cross-Dataset Validation**: UCR and UEA dataset support
- **Statistical Analysis**: Comprehensive performance metrics and efficiency rankings
- **Publication-Ready Results**: Detailed computational analysis for research papers



## Quick Start

### **1. Environment Setup**

```bash
# Clone the repository
git clone https://github.com/aminjalali-research/TSlib.git
cd TSlib

# Create conda environment
conda env create -f environment.yml
conda activate tslib

# Install additional model-specific environments
conda env create -f models/vq_mtm/environment_vq_mtm.yml
conda env create -f models/timesurl/environment_timesurl.yml
```

### **2. Enhanced Metrics Collection (Recommended)**

#### **Single Model Analysis**
```bash
# Champion model with comprehensive metrics
python enhanced_metrics/enhanced_single_model_runner.py TimesURL Chinatown 120

# Efficiency analysis
python enhanced_metrics/enhanced_single_model_runner.py Ti_MAE AtrialFibrillation 180
python enhanced_metrics/enhanced_single_model_runner.py MF_CLR AtrialFibrillation 60
```

#### **Batch Model Comparison**
```bash
# Compare top performers across datasets
python enhanced_metrics/enhanced_batch_runner.py --models TimesURL,Ti_MAE,TimeHUT --datasets Chinatown,AtrialFibrillation --timeout 200

# Comprehensive family analysis
python enhanced_metrics/enhanced_batch_runner.py --models TS2vec,TimeHUT,SoftCLT --datasets Chinatown --timeout 120
```

### **3. Basic Benchmarking**
```bash
# Traditional benchmarking (basic metrics)
python unified/master_benchmark_pipeline.py --models TimesURL --datasets Chinatown --optimization --timeout 120
```

##  **Repository Structure**

```
TSlib/
├──  enhanced_metrics/           #  Primary analysis system
│   ├── enhanced_single_model_runner.py    # Individual model analysis  
│   ├── enhanced_batch_runner.py           # Batch processing
│   ├── demo.py                            # System demonstration
│   ├── results/                           # Individual results
│   └── batch_results/                     # Batch analysis summaries
├──  core/                       # Core TSlib functionality
│   ├── backbone.py                # Model architectures
│   ├── losses.py                  # Loss functions
│   ├── train.py                   # Training pipeline
│   └── unified_dataset_reader.py  # Dataset handling
├──  models/                     # Model implementations (working models only)
│   ├── timesurl/                  # 98.54% accuracy champion
│   ├── ts2vec/                    # Universal performer family
│   ├── timehut/                   # Cross-dataset consistency
│   ├── softclt/                   # UCR specialist
│   └── vq_mtm/                    # VQ-MTM model family
├──  unified/                    # Unified benchmarking system
│   ├── master_benchmark_pipeline.py      # Main pipeline
│   ├── TaskGuide.md                      # Complete usage guide
│   └── consolidated_metrics_interface.py # Legacy metrics interface
├──  results/                    # Comprehensive results
│   ├── COMPLETE_RESULTS_SUMMARY.csv      # All model results
│   ├── comprehensive_model_comparison_metrics.json
│   └── comprehensive_metrics/             # Detailed analyses
├── ⚙ setup/                      # Environment and data setup
│   ├── setup.sh                  # Main setup script  
│   ├── download_datasets.sh      # Dataset download
│   └── run_gpu.sh                 # GPU execution helper
├──  environment.yml             # Main conda environment
├──  unified/TaskGuide.md        # Complete usage documentation
└──  PROJECT_COMPLETION_SUMMARY.md      # Project status
```

### **Computational Analysis**
- **FLOPs Estimation**: Architecture-based computational complexity analysis
- **Memory Profiling**: Peak and average GPU memory usage tracking
- **Efficiency Metrics**: Accuracy per GFLOP, accuracy per GB, time efficiency
- **Energy Estimation**: Sustainability analysis with kWh consumption estimates

##  **Documentation**

- **[Complete TaskGuide](unified/TaskGuide.md)**: Comprehensive usage instructions with all working commands
- **[Project Summary](PROJECT_COMPLETION_SUMMARY.md)**: Development history and achievements
- **[Enhanced Metrics README](enhanced_metrics/README.md)**: Detailed enhanced metrics documentation


##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Citation

If you use TSlib in your research, please cite:

```bibtex
@misc{tslib2025,
  title={TSlib: Comprehensive Time Series Self-Supervised Learning Benchmark with Enhanced Computational Analysis},
  author={Amin Jalali},
  year={2025},
  howpublished={\\url{https://github.com/aminjalali-research/TSlib}}
}

**Star this repository if you find TSlib useful for your time series research!**
