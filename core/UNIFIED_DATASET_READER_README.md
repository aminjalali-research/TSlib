# Unified Dataset Reader for TSLib

A comprehensive, unified dataset reader that supports all dataset formats used across different models in the TSLib workspace.

## Overview

The Unified Dataset Reader provides a single interface to load all types of time series datasets used in TSLib, including:

- **UCR Archive** (128+ univariate time series datasets)
- **UEA Archive** (30+ multivariate time series datasets) 
- **Forecasting datasets** (ETT, Weather, Exchange rates, etc.)
- **Medical datasets** (EEG, sleep data, etc.)
- **Custom datasets** in various formats (CSV, NPY, HDF5, Pickle, etc.)

## Features

✅ **Auto-detection** of dataset types and formats  
✅ **Unified interface** for all dataset loading  
✅ **Automatic normalization** with customizable options  
✅ **Missing value handling** with mask generation  
✅ **Time feature extraction** for temporal datasets  
✅ **Memory efficient** loading for large datasets  
✅ **Extensible** architecture for new dataset types  
✅ **Comprehensive error handling** and validation  

## Supported Dataset Types

| Type | Format | Description | Example Datasets |
|------|--------|-------------|------------------|
| UCR | TSV | Univariate time series classification | Coffee, Adiac, ECG200 |
| UEA | ARFF/NPY | Multivariate time series classification | Epilepsy, Cricket, Heartbeat |
| CSV | CSV | Time series with datetime index | ETTh1, Weather, Traffic |
| NPY | NumPy | Pre-processed numpy arrays | Custom datasets |
| HDF5 | H5 | Large-scale time series data | Exchange rates, Wind |
| Pickle | PKL | Anomaly detection datasets | Custom anomaly data |
| PyTorch | PT | PyTorch tensor datasets | Pre-processed tensors |
| Feather | Feather | Pandas feather format | Fast I/O datasets |

## Quick Start

### Basic Usage

```python
from core.unified_dataset_reader import UnifiedDatasetReader

# Initialize reader
reader = UnifiedDatasetReader("datasets")

# Auto-detect and load any dataset
data = reader.load_dataset("Coffee")  # Auto-detects as UCR
print(f"Train data shape: {data['train_data'].shape}")
print(f"Number of classes: {data['num_classes']}")
```

### Specific Dataset Types

```python
# Load UCR dataset explicitly
ucr_data = reader.load_dataset("Coffee", dataset_type='ucr', task='classification')

# Load UEA dataset explicitly  
uea_data = reader.load_dataset("Epilepsy", dataset_type='uea', task='classification')

# Load forecasting dataset
forecast_data = reader.load_dataset("ETTh1", dataset_type='csv', task='forecasting')
```

### Advanced Options

```python
# Load with specific preprocessing options
data = reader.load_dataset(
    "Coffee",
    dataset_type='ucr',
    task='classification',
    normalize=True,              # Apply normalization
    add_time_features=True,      # Add temporal features
    missing_ratio=0.1           # Simulate 10% missing values
)
```

### Convenience Functions

```python
from core.unified_dataset_reader import load_ucr_dataset, load_uea_dataset

# Quick UCR loading
ucr_data = load_ucr_dataset("Coffee", normalize=True)

# Quick UEA loading
uea_data = load_uea_dataset("Epilepsy", normalize=True)

# List all available datasets
from core.unified_dataset_reader import list_all_datasets
datasets = list_all_datasets("datasets")
```

## API Reference

### UnifiedDatasetReader Class

#### Constructor
```python
UnifiedDatasetReader(dataset_root="datasets")
```

**Parameters:**
- `dataset_root` (str): Path to the datasets directory

#### Main Methods

##### `load_dataset(dataset_name, dataset_type='auto', task='classification', **kwargs)`

**Parameters:**
- `dataset_name` (str): Name of the dataset to load
- `dataset_type` (str): Type of dataset ('auto', 'ucr', 'uea', 'csv', 'npy', 'h5', 'pkl', 'pt', 'feather')
- `task` (str): Task type ('classification', 'forecasting', 'anomaly_detection', 'imputation')
- `normalize` (bool): Whether to normalize the data (default: True)
- `add_time_features` (bool): Whether to add temporal features (default: False)
- `missing_ratio` (float): Ratio of values to simulate as missing (default: 0.0)

**Returns:**
Dictionary containing:
- `train_data`: Training data array
- `train_labels`: Training labels
- `test_data`: Test data array  
- `test_labels`: Test labels
- `train_mask`/`test_mask`: Missing value masks
- `num_classes`: Number of classes
- `num_channels`: Number of channels/features
- `sequence_length`: Length of time series
- `dataset_type`: Detected/specified dataset type
- `task`: Task type

##### `list_available_datasets()`

**Returns:**
Dictionary mapping dataset types to lists of available datasets.

## Dataset-Specific Details

### UCR Datasets
- **Format**: Tab-separated values (TSV)
- **Structure**: First column is label, remaining columns are time series values
- **Shape**: (N, T, 1) after loading
- **Normalization**: Applied to specific datasets that need it
- **Labels**: Automatically mapped to {0, 1, ..., L-1}

### UEA Datasets  
- **Format**: ARFF or NPY files
- **Structure**: Multivariate time series with class labels
- **Shape**: (N, T, C) where C > 1
- **Normalization**: StandardScaler applied by default
- **Labels**: String labels mapped to integers

### Forecasting Datasets
- **Format**: CSV with datetime index
- **Structure**: Multivariate time series for prediction
- **Splits**: 60% train, 20% validation, 20% test
- **Time Features**: Hour, day, month, etc. automatically extracted
- **Normalization**: StandardScaler fitted on training data

## Error Handling

The reader includes comprehensive error handling:

```python
try:
    data = reader.load_dataset("NonExistentDataset")
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Extending the Reader

To add support for new dataset formats:

```python
class CustomUnifiedReader(UnifiedDatasetReader):
    def __init__(self, dataset_root):
        super().__init__(dataset_root)
        self.supported_formats['my_format'] = self._load_my_format
    
    def _load_my_format(self, dataset_name, **kwargs):
        # Custom loading logic
        return {
            'train_data': train_data,
            'train_labels': train_labels,
            # ... other required fields
        }
```

## Performance Tips

1. **Memory Usage**: Large datasets are loaded entirely into memory. Consider using generators for very large datasets.

2. **Normalization**: Normalization is computed once and cached. Reuse the same reader instance for multiple loads.

3. **File I/O**: The reader caches file existence checks. Create one reader instance per session.

4. **Missing Values**: Generate masks only when needed by setting `missing_ratio > 0`.

## Integration with Existing Models

The unified reader is designed to work with all existing TSLib models:

```python
# For TS2Vec
data = reader.load_dataset("Coffee", dataset_type='ucr')
train_data = data['train_data']
train_labels = data['train_labels']

# For LEAD (EEG data)
eeg_data = reader.load_dataset("EEGDataset", task='classification')

# For forecasting models
forecast_data = reader.load_dataset("ETTh1", task='forecasting')
train_slice = forecast_data['train_slice']
```

## Examples

See `examples/unified_dataset_reader_examples.py` for comprehensive usage examples including:

- Loading different dataset types
- Auto-detection demonstration
- Advanced preprocessing options
- Error handling examples
- Integration patterns

## Requirements

```python
numpy>=1.19.0
pandas>=1.0.0
scikit-learn>=0.24.0
scipy>=1.6.0
torch>=1.8.0  # Optional, for PyTorch datasets
h5py>=3.0.0   # Optional, for HDF5 datasets
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check dataset path and file structure
2. **Import errors**: Install required dependencies 
3. **Memory errors**: Use smaller datasets or implement streaming
4. **Format errors**: Verify dataset format matches expected structure

### Dataset Structure Requirements

**UCR Format:**
```
datasets/UCR/Coffee/
├── Coffee_TRAIN.tsv
└── Coffee_TEST.tsv
```

**UEA Format:**
```
datasets/UEA/Epilepsy/
├── Epilepsy_TRAIN.arff
├── Epilepsy_TEST.arff
└── Epilepsy.npy  # Alternative format
```

**CSV Format:**
```
datasets/
└── ETTh1.csv  # With 'date' column as index
```

## Contributing

To contribute new dataset support or improvements:

1. Add loading function following the `_load_*` pattern
2. Update `supported_formats` dictionary
3. Add comprehensive error handling
4. Include tests and documentation
5. Update this README

## License

This unified dataset reader is part of the TSLib project and follows the same licensing terms.
