```
https://github.com/haobinlaosi/TDBCL
https://github.com/DL4mHealth/Medformer (APAVA, TDBrain, ADFTD, PTB, PTB-XL datasets, 10 models Transformers)
https://github.com/chenguolin/NuTime  (UCR, UEA, SleepEDF, Epilepsy, etc. NuTime/src/data/preprocess.py
https://github.com/HaokunGUI/VQ_MTM  (VQ_MTM/models/    VQ_MTM/data_provider/data_factory.py   TUSZ dataset)
https://github.com/junwoopark92/Self-Supervised-Contrastive-Forecsating  ETTh1|ETTh2|ETTm1|ETTm2|Electricity|Traffic|Weather|Excange|Illness 10 models)
https://github.com/BorealisAI/ssl-for-timeseries (128 UCR, 30 UEA, 3 ETT datasets, Electricity, KPI dataset, knows how to submit jobs to slurm)
https://github.com/findalexli/TimeseriesContrastiveModels (CLOCS Mixing-up SimCLR TS-TCC	TS2Vec TFC   SleepEEG/Epilepsy/FD-A//FD-B/HAR/Gesture/ECG/EMG)
https://github.com/DL4mHealth/SLOTS (SLOTS/Mixing-up/SimCLR/TS-TCC/TS2Vec/TFC       DEAP/SEED/EPILEPSY/HAR/P19)  
https://github.com/DL4mHealth/LEAD/ (GREAT)
https://github.com/JiaW6122/PLanTS/

https://github.com/lanxiang1017/DynamicBadPairMining_ICLR24
https://github.com/blacksnail789521/TimeDRL
https://github.com/LiuZH-19/CTRL (128 UCR, 30 UEA, 3 ETT datasets, Exchange, Wind, ILI, Weather)
https://github.com/yurui12138/TS-DRC
https://github.com/theabrusch/Multiview_TS_SSL
https://github.com/maxxu05/relcon (The Opportunity, PAMAP2, HHAR, and Motionsense datasets 

https://github.com/TsingZ0/FedTGP
https://github.com/duyngtr16061999/KDMCSE
https://github.com/sfi-norwai/eMargin
https://github.com/yingxiatang/FreConvNet
```
```
TimeHUT (AtrialFibrillation)
train_unified_comprehensive.py AtrialFibrillation optimized_params --loader UEA --scenario amc_temp --amc-instance 2.04 --amc-temporal 0.08 --amc-margin 0.67 --min-tau 0.26 --max-tau 0.68 --t-max 49 --epochs 200 --verbose (18.79s, .4667)

train_unified_comprehensive.py Chinatown scheduler_exponential --loader UCR --scenario amc_temp --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 --min-tau 0.05 --max-tau 0.76 --t-max 25 --temp-method exponential --batch-size 8 --epochs 200 --verbose

For TimeHUT ablation:
python timehut_comprehensive_ablation_runner.py --dataset Chinatown --enable-gpu-monitoring
/home/amin/anaconda3/envs/tslib/bin/python timehut_comprehensive_ablation_runner.py --dataset AtrialFibrillation --enable-gpu-monitoring

python compute_enhanced_timehut_ablation_runner.py --dataset Chinatown --enable-gpu-profiling --enable-flops-counting   (output:efficiency_summary_Chinatown_20250828_201625.csv)

Running all models: 
source activate tslib && python enhanced_metrics/all_models_runner.py --models TimeHUT_Top1,TimeHUT_Top2,TimeHUT_Top3,TS2vec,TimesURL,SoftCLT,CoST,CPC,TFC,TS_TCC,TLoss,TNC,MF_CLR --datasets Chinatown --timeout 300

conda activate tslib && python enhanced_metrics/enhanced_batch_runner.py --models BIOT,Ti_MAE,SimMTM,TFC,TimeHUT,VQ_MTM,MF_CLR,DCRNN,TS2vec,CoST,TS_TCC,TLoss,TimesURL,TNC --datasets AtrialFibrillation --timeout 200
```

Our model:
python timehut_comprehensive_ablation_runner.py --scenarios "AMC_Temperature_Cosine_AlgoOptim" "AMC_Temperature_MultiCycleCosine_AlgoOptim" "AMC_Temperature_MomentumAdaptive_AlgoOptim" --epochs 200

PyHopper Strategy for TimeHUT:
python enhanced_metrics/timehut_comprehensive_ablation_runner.py  --dataset AtrialFibrillation --enable-gpu-monitoring  --scenario AMC_PyHopper_BEST_46_67  --epochs 300


Dataset
         TimeHUT TS2Vec   TNC   TS-TCC  T-Loss   TST  TF-C
AF       0.53   0.200    0.133   0.267  0.200   0.067  0.200

Use the efficient version of TimeHUT, less flops and gpu memory, use optimized configuration for both AMC, then from that initial config optimize all param including each scheduler params



Practice talking frameworks: 1 thing, 2 types, 3 steps
They consider you as trainer!!! Damn frustrating!

run  each model and part in a separated file and put a proper name for it, use "edit" mode not "agent". solve one problem at the time. Never "agent" mode.

- no emoji and give root address instead of using my name
- prepare a chechlist and alway follow it
- Create doc folder (all documents), .github (all the initializations), separate files for documentation (put correct names for them and when you want to reuse does not occupy your window context)
how to code that does not destroy the other parts?
Docs folder: Bug_tracking.md, Implementation.md, Project_structure.md, UI_UX_doc.md




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
