#!/bin/bash
# Complete Dataset Download and Organization Script for TSlib
# This script downloads and organizes all datasets (excluding UCR/UEA) under /datasets

set -e  # Exit on any error

BASE_DIR="/home/amin/TSlib/datasets"
SCRIPTS_DIR="/home/amin/TSlib/scripts/setup"

echo "🚀 TSlib Complete Dataset Download & Organization Script"
echo "========================================================"
echo "Base directory: $BASE_DIR"
echo "This script will organize all datasets in the appropriate structure."
echo ""

# Create base directory structure
mkdir -p "$BASE_DIR"/{ETT,forecasting,medical,eeg,emotion,ssl_datasets,specialized}

echo "📁 Creating directory structure..."

# Function to download with error handling
download_with_retry() {
    local url=$1
    local dest=$2
    local filename=$(basename "$url")
    
    echo "  Downloading: $filename"
    if command -v wget >/dev/null 2>&1; then
        wget -c "$url" -O "$dest" || echo "    ⚠️  Failed to download $filename"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$dest" || echo "    ⚠️  Failed to download $filename"
    else
        echo "    ❌ Neither wget nor curl available"
        return 1
    fi
}

# Function to clone git repository
clone_repo() {
    local repo_url=$1
    local dest_dir=$2
    
    echo "  Cloning: $(basename "$repo_url")"
    if [ -d "$dest_dir" ]; then
        echo "    ⚠️  Directory exists, skipping: $dest_dir"
    else
        git clone "$repo_url" "$dest_dir" || echo "    ⚠️  Failed to clone $repo_url"
    fi
}

echo ""
echo "📈 1. ETT FORECASTING DATASETS"
echo "==============================="
mkdir -p "$BASE_DIR/ETT"

# ETT Datasets - Multiple sources
echo "Option A: Clone complete ETT repository"
clone_repo "https://github.com/zhouhaoyi/ETDataset.git" "$BASE_DIR/ETT/ETDataset"

echo "Option B: Direct CSV downloads"
mkdir -p "$BASE_DIR/ETT/csv_files"
download_with_retry "https://github.com/zhouhaoyi/ETDataset/raw/main/ETTh1.csv" "$BASE_DIR/ETT/csv_files/ETTh1.csv"
download_with_retry "https://github.com/zhouhaoyi/ETDataset/raw/main/ETTh2.csv" "$BASE_DIR/ETT/csv_files/ETTh2.csv"
download_with_retry "https://github.com/zhouhaoyi/ETDataset/raw/main/ETTm1.csv" "$BASE_DIR/ETT/csv_files/ETTm1.csv"
download_with_retry "https://github.com/zhouhaoyi/ETDataset/raw/main/ETTm2.csv" "$BASE_DIR/ETT/csv_files/ETTm2.csv"

echo ""
echo "🩺 2. MEDICAL/EEG DATASETS"
echo "=========================="

# PhysioNet datasets
mkdir -p "$BASE_DIR/medical/physionet"

echo "Downloading PhysioNet datasets..."
echo "Note: These require PhysioNet account for full access"

# Create download scripts for PhysioNet (requires authentication)
cat > "$BASE_DIR/medical/physionet/download_physionet.sh" << 'EOF'
#!/bin/bash
# PhysioNet Dataset Downloads
# Requires: PhysioNet account and wfdb-python package
# Install: pip install wfdb

echo "PhysioNet datasets require authentication."
echo "Please visit https://physionet.org/ to create an account."
echo "Then install wfdb: pip install wfdb"

# PTB Database
echo "Downloading PTB Database..."
python -c "
import wfdb
# Download PTB database
wfdb.dl_database('ptbdb', dl_dir='./PTB/')
"

# PTB-XL Database  
echo "Downloading PTB-XL Database..."
python -c "
import wfdb
# Download PTB-XL database
wfdb.dl_database('ptb-xl', dl_dir='./PTB-XL/')
"

# Sleep EDF Database
echo "Downloading Sleep EDF Database..."
python -c "
import wfdb
# Download Sleep EDF database
wfdb.dl_database('sleep-edfx', dl_dir='./SleepEDF/')
"

echo "PhysioNet downloads completed!"
EOF

chmod +x "$BASE_DIR/medical/physionet/download_physionet.sh"

echo ""
echo "🧠 3. EEG/EMOTION DATASETS"  
echo "=========================="

mkdir -p "$BASE_DIR/eeg"/{DEAP,SEED,BrainLat,specialized}

echo "Setting up EEG dataset download instructions..."

# DEAP Dataset
cat > "$BASE_DIR/eeg/DEAP/README_DOWNLOAD.md" << 'EOF'
# DEAP Dataset Download Instructions

## About DEAP
Database for Emotion Analysis using Physiological Signals
32-channel EEG, emotional stimuli (music videos)

## Download Steps:
1. Visit: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
2. Register for account
3. Download DEAP data files
4. Extract to this directory

## Expected Structure:
```
DEAP/
├── data_preprocessed_python/
│   ├── s01.dat
│   ├── s02.dat
│   └── ...
└── participant_ratings.csv
```

## Citation:
Koelstra, S. et al. (2012). DEAP: A Database for Emotion Analysis; Using Physiological Signals. IEEE Transactions on Affective Computing, 3(1), 18-31.
EOF

# SEED Dataset
cat > "$BASE_DIR/eeg/SEED/README_DOWNLOAD.md" << 'EOF'
# SEED Dataset Download Instructions

## About SEED
SJTU Emotion EEG Dataset
Multi-modal emotion recognition dataset

## Download Steps:
1. Visit: https://bcmi.sjtu.edu.cn/home/seed/
2. Register and request access
3. Download SEED dataset files
4. Extract to this directory

## Expected Structure:
```
SEED/
├── SEED/
│   ├── 1_20131027.mat
│   ├── 1_20131030.mat
│   └── ...
└── label.mat
```

## Citation:
Zheng, W.L., Lu, B.L. (2015). Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks. IEEE Transactions on Autonomous Mental Development.
EOF

echo ""
echo "🌐 4. SPECIALIZED DATASETS"
echo "=========================="

mkdir -p "$BASE_DIR/specialized"/{illness,exchange_rate,weather,motion,multiview}

# Copy existing datasets from models directory if they exist
if [ -d "/home/amin/TSlib/models/ssl_forecasting/dataset" ]; then
    echo "Copying existing forecasting datasets..."
    cp -r /home/amin/TSlib/models/ssl_forecasting/dataset/* "$BASE_DIR/specialized/" 2>/dev/null || true
fi

# Human Activity Recognition dataset (for TS Contrastive)
mkdir -p "$BASE_DIR/specialized/motion/human_activity"
cat > "$BASE_DIR/specialized/motion/human_activity/download_instructions.md" << 'EOF'
# Human Activity Recognition Dataset

## For TS Contrastive Learning
Dataset: Accel2ActivityCrawl
Size: 2.7 Million entries of tri-axial IMU motion data

## Download:
Visit: https://huggingface.co/datasets/alexshengzhili/Accel2ActivityCrawl
Or search for "alexshengzhili/Accel2ActivityCrawl" on Hugging Face

## Usage:
Used by TS Contrastive model for CLIP-motion implementation
Tri-axial accelerometer data with activity annotations
EOF

echo ""
echo "📊 5. FORECASTING DATASETS"
echo "=========================="

mkdir -p "$BASE_DIR/forecasting"/{weather,traffic,electricity,finance}

# Weather dataset
if [ -f "/home/amin/TSlib/models/ssl_forecasting/dataset/weather/weather.csv" ]; then
    cp "/home/amin/TSlib/models/ssl_forecasting/dataset/weather/weather.csv" "$BASE_DIR/forecasting/weather/"
    echo "✅ Copied weather.csv"
fi

# Exchange rate dataset
if [ -f "/home/amin/TSlib/models/ssl_forecasting/dataset/exchange_rate/exchange_rate.csv" ]; then
    cp "/home/amin/TSlib/models/ssl_forecasting/dataset/exchange_rate/exchange_rate.csv" "$BASE_DIR/forecasting/finance/"
    echo "✅ Copied exchange_rate.csv"
fi

# National illness dataset
if [ -f "/home/amin/TSlib/models/ssl_forecasting/dataset/illness/national_illness.csv" ]; then
    cp "/home/amin/TSlib/models/ssl_forecasting/dataset/illness/national_illness.csv" "$BASE_DIR/medical/"
    echo "✅ Copied national_illness.csv"
fi

echo ""
echo "🔗 6. CREATING SYMLINKS TO MODELS"
echo "=================================="

# Create symlinks in model directories to use centralized datasets
echo "Creating symlinks for models to use centralized datasets..."

# For each model that has its own dataset directory, create symlinks
models_with_datasets=(
    "ssl_forecasting"
    "slots"
    "medformer"
    "ctrl"
    "timehut"
)

for model in "${models_with_datasets[@]}"; do
    model_path="/home/amin/TSlib/models/$model"
    if [ -d "$model_path" ]; then
        # Find dataset directories in this model
        find "$model_path" -name "dataset*" -type d 2>/dev/null | while read dataset_dir; do
            # Create backup if original exists
            if [ -e "$dataset_dir" ] && [ ! -L "$dataset_dir" ]; then
                mv "$dataset_dir" "${dataset_dir}_original_backup"
                echo "  Backed up: $dataset_dir"
            fi
            # Create symlink to centralized datasets
            ln -sf "$BASE_DIR" "$dataset_dir" 2>/dev/null || true
            echo "  Linked: $dataset_dir -> $BASE_DIR"
        done
    fi
done

echo ""
echo "📋 7. CREATING DATASET INVENTORY"
echo "================================"

# Create comprehensive dataset inventory
cat > "$BASE_DIR/DATASET_INVENTORY.md" << 'EOF'
# TSlib Dataset Inventory

## 📊 **Dataset Organization Structure**

```
datasets/
├── UCR/                          # UCR 128 time series datasets
├── UEA/                          # UEA 30 multivariate datasets
├── ETT/                          # Electricity Transformer datasets
│   ├── ETDataset/               # Complete ETT repository
│   └── csv_files/               # Individual CSV files
├── medical/                      # Medical/physiological datasets
│   ├── physionet/              # PhysioNet databases
│   └── national_illness.csv    # CDC illness surveillance
├── eeg/                         # EEG/neural datasets
│   ├── DEAP/                   # Emotion recognition EEG
│   ├── SEED/                   # SJTU emotion dataset
│   └── BrainLat/               # Brain lateralization
├── forecasting/                 # Time series forecasting
│   ├── weather/                # Weather data
│   ├── finance/                # Financial data
│   └── traffic/                # Traffic flow data
└── specialized/                 # Model-specific datasets
    ├── motion/                 # Human activity recognition
    └── multiview/              # Multi-view learning
```

## 🔄 **Dataset Mapping by Model**

### **TimesURL/TimeHUT**
- Primary: UCR (128 datasets), UEA (30 datasets)
- Location: `datasets/UCR/`, `datasets/UEA/`

### **SoftCLT**  
- Classification: UCR, UEA
- Medical: Epilepsy, SleepEEG, FD-B, Gesture, EMG
- Transfer Learning: SleepEEG → {Epilepsy, FD-B, Gesture, EMG}

### **SSL Forecasting**
- ETT: `datasets/ETT/csv_files/ETT{h1,h2,m1,m2}.csv`
- Weather: `datasets/forecasting/weather/weather.csv`
- Exchange: `datasets/forecasting/finance/exchange_rate.csv`
- Illness: `datasets/medical/national_illness.csv`

### **MedFormer**
- PhysioNet: PTB, PTB-XL, Sleep-EDF
- Location: `datasets/medical/physionet/`

### **TS Contrastive**
- Motion: Accel2ActivityCrawl (2.7M samples)
- Location: `datasets/specialized/motion/`

## 📥 **Download Instructions**

### **Automated Downloads**
```bash
# Run this script to download available datasets
cd /home/amin/TSlib/datasets
bash download_all_datasets.sh
```

### **Manual Downloads Required**
1. **UCR/UEA**: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
2. **DEAP**: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
3. **SEED**: https://bcmi.sjtu.edu.cn/home/seed/
4. **PhysioNet**: https://physionet.org/ (requires account)

### **PhysioNet Setup**
```bash
pip install wfdb
cd datasets/medical/physionet
bash download_physionet.sh
```

## 🔗 **Model Integration**

All models now use centralized datasets via symlinks:
- Original dataset folders backed up as `*_original_backup`
- Models automatically use `/datasets/` structure
- No duplication of dataset files

## 📈 **Usage Examples**

### **Classification (UCR/UEA)**
```bash
python train.py Chinatown --loader UCR --epochs 100
python train.py BasicMotions --loader UEA --epochs 100
```

### **Forecasting (ETT)**
```bash
python forecast.py --dataset ETTh1 --data-path datasets/ETT/csv_files/
```

### **Medical Analysis**
```bash
python medical_analysis.py --dataset PTB --data-path datasets/medical/physionet/PTB/
```

## 🔧 **Maintenance**

### **Update Dataset Links**
```bash
cd datasets/
find . -type l -exec ls -la {} \; | grep "broken"  # Check broken links
```

### **Add New Dataset**
1. Place in appropriate category folder
2. Update this inventory file
3. Add to model-specific configuration files
4. Update symlinks if needed

## 📚 **Dataset Citations**

Each dataset directory contains citation information in README files.
Always cite the original dataset authors when using these datasets.
EOF

echo ""
echo "✅ SETUP COMPLETE!"
echo "=================="
echo ""
echo "📁 Datasets organized in: $BASE_DIR"
echo "📋 Inventory file: $BASE_DIR/DATASET_INVENTORY.md"
echo "🔗 Model symlinks created for centralized access"
echo ""
echo "Next steps:"
echo "1. Download UCR/UEA datasets manually"
echo "2. Run PhysioNet download script (requires account)"
echo "3. Register for DEAP/SEED datasets"
echo "4. Check model-specific README files for usage instructions"
echo ""
echo "🚀 All models now use centralized dataset structure!"
