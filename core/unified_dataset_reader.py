"""
Unified Dataset Reader for TSLib
Supports loading all dataset types used across different models in the TSLib workspace.
"""

import os
import numpy as np
import pandas as pd
import h5py
import pickle
import torch
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple, Optional, Union, List
import warnings
from pathlib import Path
import time
from datetime import datetime

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class UnifiedDatasetReader:
    """
    Unified dataset reader that supports all datasets used across TSLib models.
    
    Supported dataset types:
    - UCR Archive (univariate time series)
    - UEA Archive (multivariate time series)
    - ETT datasets (forecasting)
    - Custom CSV datasets (forecasting)
    - HDF5 datasets (forecasting)
    - NPY datasets (various tasks)
    - EEG datasets (medical/brain signals)
    - Custom pickle datasets
    - ARFF datasets
    """
    
    def __init__(self, dataset_root: str = "datasets"):
        self.dataset_root = Path(dataset_root)
        self.supported_formats = {
            'ucr': self._load_ucr,
            'uea': self._load_uea, 
            'csv': self._load_csv,
            'npy': self._load_npy,
            'h5': self._load_h5,
            'pkl': self._load_pickle,
            'arff': self._load_arff,
            'eeg': self._load_eeg,
            'pt': self._load_pytorch,
            'feather': self._load_feather
        }
        
        # Normalization datasets for UCR
        self.ucr_non_normalized_datasets = {
            'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
            'BME', 'Chinatown', 'Crop', 'EOGHorizontalSignal', 'EOGVerticalSignal',
            'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
            'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPointAgeSpan',
            'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'HouseTwenty',
            'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'MelbournePedestrian',
            'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure',
            'PigCVP', 'PLAID', 'PowerCons', 'Rock', 'SemgHandGenderCh2',
            'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
            'SmoothSubspace', 'UMD'
        }
        
    def load_dataset(self, 
                    dataset_name: str,
                    dataset_type: str = 'auto',
                    task: str = 'classification',
                    normalize: bool = True,
                    add_time_features: bool = False,
                    **kwargs) -> Dict:
        """
        Load a dataset with automatic format detection or manual specification.
        
        Args:
            dataset_name: Name of the dataset
            dataset_type: Type of dataset ('auto', 'ucr', 'uea', 'csv', 'npy', etc.)
            task: Task type ('classification', 'forecasting', 'anomaly_detection', 'imputation')
            normalize: Whether to normalize the data
            add_time_features: Whether to add temporal features
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Dictionary containing data, labels, and metadata
        """
        
        # Auto-detect dataset type if not specified
        if dataset_type == 'auto':
            dataset_type = self._detect_dataset_type(dataset_name)
            
        # Load dataset using appropriate loader
        if dataset_type not in self.supported_formats:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
        loader_func = self.supported_formats[dataset_type]
        return loader_func(dataset_name, task=task, normalize=normalize, 
                         add_time_features=add_time_features, **kwargs)
    
    def _detect_dataset_type(self, dataset_name: str) -> str:
        """Auto-detect dataset type based on available files."""
        
        # Check for UCR format
        ucr_path = self.dataset_root / "UCR" / dataset_name
        if ucr_path.exists():
            train_file = ucr_path / f"{dataset_name}_TRAIN.tsv"
            if train_file.exists():
                return 'ucr'
                
        # Check for UEA format  
        uea_path = self.dataset_root / "UEA" / dataset_name
        if uea_path.exists():
            arff_file = uea_path / f"{dataset_name}_TRAIN.arff"
            npy_file = uea_path / f"{dataset_name}.npy"
            if arff_file.exists() or npy_file.exists():
                return 'uea'
        
        # Check for common forecasting datasets
        csv_file = self.dataset_root / f"{dataset_name}.csv"
        if csv_file.exists():
            return 'csv'
            
        # Check for numpy files
        npy_file = self.dataset_root / f"{dataset_name}.npy"
        if npy_file.exists():
            return 'npy'
            
        # Check for HDF5 files
        h5_file = self.dataset_root / f"{dataset_name}.h5"
        if h5_file.exists():
            return 'h5'
            
        # Check for pickle files
        pkl_file = self.dataset_root / f"{dataset_name}.pkl"
        if pkl_file.exists():
            return 'pkl'
            
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {self.dataset_root}")
    
    def _load_ucr(self, dataset_name: str, task: str = 'classification', 
                  normalize: bool = True, add_time_features: bool = False, **kwargs) -> Dict:
        """Load UCR univariate time series dataset."""
        
        dataset_path = self.dataset_root / "UCR" / dataset_name
        train_file = dataset_path / f"{dataset_name}_TRAIN.tsv"
        test_file = dataset_path / f"{dataset_name}_TEST.tsv"
        
        if not (train_file.exists() and test_file.exists()):
            raise FileNotFoundError(f"UCR dataset files not found for {dataset_name}")
            
        # Load data
        train_df = pd.read_csv(train_file, sep='\t', header=None)
        test_df = pd.read_csv(test_file, sep='\t', header=None)
        
        train_array = np.array(train_df)
        test_array = np.array(test_df)
        
        # Extract labels and transform to {0, ..., L-1}
        labels = np.unique(train_array[:, 0])
        transform = {l: i for i, l in enumerate(labels)}
        
        train_data = train_array[:, 1:].astype(np.float64)
        train_labels = np.vectorize(transform.get)(train_array[:, 0])
        test_data = test_array[:, 1:].astype(np.float64)
        test_labels = np.vectorize(transform.get)(test_array[:, 0])
        
        # Handle NaN values for specific datasets
        if dataset_name in ['PickupGestureWiimoteZ']:
            train_data = np.nan_to_num(train_data, nan=0.0)
            test_data = np.nan_to_num(test_data, nan=0.0)
        
        # Add channel dimension (N, T, C)
        train_data = train_data[..., np.newaxis]
        test_data = test_data[..., np.newaxis]
        
        # Generate masks
        train_mask = self._generate_mask(train_data, missing_ratio=kwargs.get('missing_ratio', 0.0))
        test_mask = self._generate_mask(test_data, missing_ratio=kwargs.get('missing_ratio', 0.0))
        
        # Normalization
        if normalize and dataset_name in self.ucr_non_normalized_datasets:
            scaler = StandardScaler()
            train_data, test_data = self._normalize_with_mask(
                train_data, train_mask, test_data, test_mask, scaler
            )
        
        # Add time features if requested
        if add_time_features:
            time_features = np.linspace(0, 1, train_data.shape[1]).reshape(1, -1, 1)
            train_time = np.repeat(time_features, train_data.shape[0], axis=0)
            test_time = np.repeat(time_features, test_data.shape[0], axis=0)
            train_data = np.concatenate([train_data, train_time], axis=-1)
            test_data = np.concatenate([test_data, test_time], axis=-1)
        
        return {
            'train_data': train_data,
            'train_labels': train_labels,
            'train_mask': train_mask,
            'test_data': test_data,
            'test_labels': test_labels,
            'test_mask': test_mask,
            'num_classes': len(labels),
            'num_channels': train_data.shape[-1],
            'sequence_length': train_data.shape[1],
            'dataset_type': 'ucr',
            'task': task
        }
    
    def _load_uea(self, dataset_name: str, task: str = 'classification',
                  normalize: bool = True, add_time_features: bool = False, **kwargs) -> Dict:
        """Load UEA multivariate time series dataset."""
        
        dataset_path = self.dataset_root / "UEA" / dataset_name
        
        # Try loading ARFF files first
        try:
            train_file = dataset_path / f"{dataset_name}_TRAIN.arff"
            test_file = dataset_path / f"{dataset_name}_TEST.arff"
            
            train_data_raw = loadarff(str(train_file))[0]
            test_data_raw = loadarff(str(test_file))[0]
            
            train_data, train_labels = self._extract_arff_data(train_data_raw)
            test_data, test_labels = self._extract_arff_data(test_data_raw)
            
        except:
            # Try loading numpy files
            npy_file = dataset_path / f"{dataset_name}.npy"
            if npy_file.exists():
                data = np.load(str(npy_file), allow_pickle=True).item()
                train_data = data["train_X"]
                train_labels = data["train_y"]
                test_data = data["test_X"]
                test_labels = data["test_y"]
            else:
                raise FileNotFoundError(f"UEA dataset files not found for {dataset_name}")
        
        # Generate masks
        train_mask = self._generate_mask(train_data, missing_ratio=kwargs.get('missing_ratio', 0.0))
        test_mask = self._generate_mask(test_data, missing_ratio=kwargs.get('missing_ratio', 0.0))
        
        # Normalization
        if normalize:
            scaler = StandardScaler()
            train_data, test_data = self._normalize_with_mask(
                train_data, train_mask, test_data, test_mask, scaler
            )
        
        # Add time features if requested
        if add_time_features:
            time_features = np.linspace(0, 1, train_data.shape[1]).reshape(1, -1, 1)
            train_time = np.repeat(time_features, train_data.shape[0], axis=0)
            test_time = np.repeat(time_features, test_data.shape[0], axis=0)
            train_data = np.concatenate([train_data, train_time], axis=-1)
            test_data = np.concatenate([test_data, test_time], axis=-1)
        
        # Transform labels
        labels = np.unique(train_labels)
        transform = {l: i for i, l in enumerate(labels)}
        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)
        
        return {
            'train_data': train_data,
            'train_labels': train_labels,
            'train_mask': train_mask,
            'test_data': test_data,
            'test_labels': test_labels,
            'test_mask': test_mask,
            'num_classes': len(labels),
            'num_channels': train_data.shape[-1],
            'sequence_length': train_data.shape[1],
            'dataset_type': 'uea',
            'task': task
        }
    
    def _load_csv(self, dataset_name: str, task: str = 'forecasting',
                  normalize: bool = True, add_time_features: bool = True, **kwargs) -> Dict:
        """Load CSV datasets (typically for forecasting tasks)."""
        
        csv_file = self.dataset_root / f"{dataset_name}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Load data with date parsing if date column exists
        try:
            data = pd.read_csv(csv_file, index_col='date', parse_dates=True)
        except:
            data = pd.read_csv(csv_file)
            
        # Handle different dataset formats
        if task == 'forecasting':
            return self._process_forecasting_csv(data, dataset_name, normalize, add_time_features, **kwargs)
        else:
            return self._process_general_csv(data, dataset_name, normalize, **kwargs)
    
    def _load_npy(self, dataset_name: str, task: str = 'auto',
                  normalize: bool = True, add_time_features: bool = False, **kwargs) -> Dict:
        """Load NPY datasets."""
        
        npy_file = self.dataset_root / f"{dataset_name}.npy"
        if not npy_file.exists():
            raise FileNotFoundError(f"NPY file not found: {npy_file}")
        
        data = np.load(str(npy_file), allow_pickle=True)
        
        # Handle different NPY formats
        if isinstance(data, np.ndarray) and data.dtype == object:
            # Dictionary format
            data = data.item()
            if 'train_X' in data:
                return self._process_structured_npy(data, dataset_name, normalize, add_time_features, **kwargs)
        
        # Simple array format for forecasting
        if task == 'forecasting' or (task == 'auto' and len(data.shape) == 2):
            return self._process_forecasting_npy(data, dataset_name, normalize, **kwargs)
        
        # Default processing
        return self._process_general_npy(data, dataset_name, normalize, **kwargs)
    
    def _load_h5(self, dataset_name: str, task: str = 'forecasting',
                 normalize: bool = True, **kwargs) -> Dict:
        """Load HDF5 datasets."""
        
        h5_file = self.dataset_root / f"{dataset_name}.h5"
        if not h5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
        
        with h5py.File(str(h5_file), 'r') as f:
            data = f['data'][:]
        
        return self._process_forecasting_npy(data, dataset_name, normalize, **kwargs)
    
    def _load_pickle(self, dataset_name: str, task: str = 'anomaly_detection',
                     normalize: bool = True, **kwargs) -> Dict:
        """Load pickle datasets (typically for anomaly detection)."""
        
        pkl_file = self.dataset_root / f"{dataset_name}.pkl"
        if not pkl_file.exists():
            raise FileNotFoundError(f"Pickle file not found: {pkl_file}")
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        return self._process_anomaly_pickle(data, dataset_name, normalize, **kwargs)
    
    def _load_arff(self, dataset_name: str, **kwargs) -> Dict:
        """Load ARFF datasets."""
        return self._load_uea(dataset_name, **kwargs)
    
    def _load_eeg(self, dataset_name: str, task: str = 'classification',
                  normalize: bool = True, **kwargs) -> Dict:
        """Load EEG datasets (specialized for brain signals)."""
        
        # This would need to be customized based on specific EEG data formats
        # For now, delegate to appropriate format
        return self.load_dataset(dataset_name, dataset_type='auto', task=task, 
                               normalize=normalize, **kwargs)
    
    def _load_pytorch(self, dataset_name: str, **kwargs) -> Dict:
        """Load PyTorch tensor datasets."""
        
        pt_file = self.dataset_root / f"{dataset_name}.pt"
        if not pt_file.exists():
            raise FileNotFoundError(f"PyTorch file not found: {pt_file}")
        
        data = torch.load(str(pt_file))
        
        return {
            'train_data': data.get('train_data', data.get('samples')),
            'train_labels': data.get('train_labels', data.get('targets')),
            'test_data': data.get('test_data'),
            'test_labels': data.get('test_labels'),
            'dataset_type': 'pytorch',
            'task': kwargs.get('task', 'classification')
        }
    
    def _load_feather(self, dataset_name: str, **kwargs) -> Dict:
        """Load Feather datasets."""
        
        feather_file = self.dataset_root / f"{dataset_name}.feather"
        if not feather_file.exists():
            raise FileNotFoundError(f"Feather file not found: {feather_file}")
        
        data = pd.read_feather(str(feather_file))
        return self._process_general_csv(data, dataset_name, **kwargs)
    
    # Helper methods
    def _extract_arff_data(self, data):
        """Extract data from ARFF format."""
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8") if isinstance(t_label, bytes) else t_label
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    def _generate_mask(self, data: np.ndarray, missing_ratio: float = 0.0) -> np.ndarray:
        """Generate mask for missing values."""
        if missing_ratio == 0.0:
            return np.ones_like(data)
        
        mask = np.ones_like(data)
        if missing_ratio > 0:
            num_missing = int(missing_ratio * data.size)
            flat_indices = np.random.choice(data.size, num_missing, replace=False)
            mask.flat[flat_indices] = 0
        
        return mask
    
    def _normalize_with_mask(self, train_data, train_mask, test_data, test_mask, scaler):
        """Normalize data considering missing values."""
        # Flatten for normalization
        train_flat = train_data[train_mask == 1]
        scaler.fit(train_flat.reshape(-1, 1))
        
        # Normalize
        train_normalized = train_data.copy()
        test_normalized = test_data.copy()
        
        train_normalized[train_mask == 1] = scaler.transform(
            train_data[train_mask == 1].reshape(-1, 1)
        ).flatten()
        
        test_normalized[test_mask == 1] = scaler.transform(
            test_data[test_mask == 1].reshape(-1, 1)
        ).flatten()
        
        return train_normalized, test_normalized
    
    def _process_forecasting_csv(self, data, dataset_name, normalize, add_time_features, **kwargs):
        """Process CSV data for forecasting tasks."""
        
        # Extract time features if timestamp index
        if hasattr(data.index, 'to_pydatetime'):
            dt_features = self._get_time_features(data.index)
        else:
            dt_features = None
        
        # Convert to numpy
        data_values = data.values
        
        # Handle specific dataset formats
        if dataset_name.lower().startswith('ett'):
            train_slice = slice(None, 12 * 30 * 24)
            valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
            test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
        else:
            train_slice = slice(None, int(0.6 * len(data_values)))
            valid_slice = slice(int(0.6 * len(data_values)), int(0.8 * len(data_values)))
            test_slice = slice(int(0.8 * len(data_values)), None)
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            scaler.fit(data_values[train_slice])
            data_values = scaler.transform(data_values)
        
        return {
            'data': data_values,
            'train_slice': train_slice,
            'valid_slice': valid_slice,
            'test_slice': test_slice,
            'time_features': dt_features,
            'scaler': scaler if normalize else None,
            'dataset_type': 'csv_forecasting',
            'task': 'forecasting'
        }
    
    def _get_time_features(self, dt_index):
        """Extract time features from datetime index."""
        return np.stack([
            dt_index.minute.to_numpy(),
            dt_index.hour.to_numpy(),
            dt_index.dayofweek.to_numpy(),
            dt_index.day.to_numpy(),
            dt_index.dayofyear.to_numpy(),
            dt_index.month.to_numpy(),
            dt_index.isocalendar().week.to_numpy(),
        ], axis=1).astype(np.float64)
    
    def _process_structured_npy(self, data, dataset_name, normalize, add_time_features, **kwargs):
        """Process structured numpy data."""
        
        result = {
            'train_data': data.get('train_X', data.get('tr_x')),
            'train_labels': data.get('train_y', data.get('tr_y')),
            'test_data': data.get('test_X', data.get('te_x')),
            'test_labels': data.get('test_y', data.get('te_y')),
            'train_mask': data.get('tr_mask'),
            'test_mask': data.get('te_mask'),
            'dataset_type': 'structured_npy',
            'task': kwargs.get('task', 'classification')
        }
        
        return result
    
    def _process_forecasting_npy(self, data, dataset_name, normalize, **kwargs):
        """Process numpy data for forecasting."""
        
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
        
        if normalize:
            scaler = StandardScaler()
            scaler.fit(data[train_slice])
            data = scaler.transform(data)
        
        return {
            'data': data,
            'train_slice': train_slice,
            'valid_slice': valid_slice,
            'test_slice': test_slice,
            'scaler': scaler if normalize else None,
            'dataset_type': 'npy_forecasting',
            'task': 'forecasting'
        }
    
    def _process_general_csv(self, data, dataset_name, normalize, **kwargs):
        """Process general CSV data."""
        
        data_values = data.values
        if data_values.shape[1] > 1:
            # Assume last column is labels
            X = data_values[:, :-1]
            y = data_values[:, -1]
        else:
            X = data_values
            y = None
        
        if normalize:
            scaler = StandardScaler()
            X = scaler.transform(X)
        
        return {
            'data': X,
            'labels': y,
            'scaler': scaler if normalize else None,
            'dataset_type': 'csv_general',
            'task': kwargs.get('task', 'classification')
        }
    
    def _process_general_npy(self, data, dataset_name, normalize, **kwargs):
        """Process general numpy data."""
        
        if normalize:
            scaler = StandardScaler()
            data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        
        return {
            'data': data,
            'scaler': scaler if normalize else None,
            'dataset_type': 'npy_general',
            'task': kwargs.get('task', 'classification')
        }
    
    def _process_anomaly_pickle(self, data, dataset_name, normalize, **kwargs):
        """Process pickle data for anomaly detection."""
        
        return {
            'train_data': data.get('all_train_data'),
            'train_labels': data.get('all_train_labels'),
            'train_timestamps': data.get('all_train_timestamps'),
            'test_data': data.get('all_test_data'),
            'test_labels': data.get('all_test_labels'),
            'test_timestamps': data.get('all_test_timestamps'),
            'delay': data.get('delay'),
            'dataset_type': 'anomaly_pickle',
            'task': 'anomaly_detection'
        }
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """List all available datasets by type."""
        
        available = {
            'ucr': [],
            'uea': [],
            'csv': [],
            'npy': [],
            'h5': [],
            'pkl': [],
            'other': []
        }
        
        # Check UCR datasets
        ucr_path = self.dataset_root / "UCR"
        if ucr_path.exists():
            available['ucr'] = [d.name for d in ucr_path.iterdir() if d.is_dir()]
        
        # Check UEA datasets  
        uea_path = self.dataset_root / "UEA"
        if uea_path.exists():
            available['uea'] = [d.name for d in uea_path.iterdir() if d.is_dir()]
        
        # Check other formats in root
        for ext in ['csv', 'npy', 'h5', 'pkl']:
            files = list(self.dataset_root.glob(f"*.{ext}"))
            available[ext] = [f.stem for f in files]
        
        return available


# Convenience functions for common use cases
def load_ucr_dataset(dataset_name: str, dataset_root: str = "datasets", **kwargs) -> Dict:
    """Convenience function to load UCR dataset."""
    reader = UnifiedDatasetReader(dataset_root)
    return reader.load_dataset(dataset_name, dataset_type='ucr', **kwargs)


def load_uea_dataset(dataset_name: str, dataset_root: str = "datasets", **kwargs) -> Dict:
    """Convenience function to load UEA dataset."""
    reader = UnifiedDatasetReader(dataset_root)
    return reader.load_dataset(dataset_name, dataset_type='uea', **kwargs)


def load_forecasting_dataset(dataset_name: str, dataset_root: str = "datasets", **kwargs) -> Dict:
    """Convenience function to load forecasting dataset."""
    reader = UnifiedDatasetReader(dataset_root)
    return reader.load_dataset(dataset_name, task='forecasting', **kwargs)


def list_all_datasets(dataset_root: str = "datasets") -> Dict[str, List[str]]:
    """List all available datasets."""
    reader = UnifiedDatasetReader(dataset_root)
    return reader.list_available_datasets()


# Example usage
if __name__ == "__main__":
    # Initialize reader
    reader = UnifiedDatasetReader("datasets")
    
    # List available datasets
    datasets = reader.list_available_datasets()
    print("Available datasets:", datasets)
    
    # Load a UCR dataset
    try:
        ucr_data = reader.load_dataset("Coffee", dataset_type='ucr', task='classification')
        print(f"Loaded Coffee dataset: {ucr_data['train_data'].shape}")
    except:
        print("Coffee dataset not found")
    
    # Load a UEA dataset
    try:
        uea_data = reader.load_dataset("Epilepsy", dataset_type='uea', task='classification')
        print(f"Loaded Epilepsy dataset: {uea_data['train_data'].shape}")
    except:
        print("Epilepsy dataset not found")
