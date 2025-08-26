#!/usr/bin/env python3
"""
Enhanced Batch Model Runner
===========================

Batch runner for collecting enhanced metrics (Time/Epoch, Peak GPU Memory, FLOPs/Epoch) 
across multiple models and datasets systematically.

This companion script to enhanced_single_model_runner.py enables:
✅ Batch processing of model-dataset combinations
✅ Comprehensive enhanced metrics collection
✅ Progress tracking and error handling
✅ Consolidated reporting across all runs
✅ Time estimation and resource planning

Usage:
    python enhanced_metrics/enhanced_batch_runner.py --config <config_file>
    python enhanced_metrics/enhanced_batch_runner.py --models TimesURL,BIOT --datasets Chinatown,AtrialFibrillation
"""

import json
import time
import sys
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import subprocess

# Import the single model runner
sys.path.append(str(Path(__file__).parent))
from enhanced_single_model_runner import EnhancedSingleModelRunner


class EnhancedBatchRunner:
    """Batch runner for enhanced metrics collection across multiple models and datasets"""
    
    def __init__(self):
        self.results_dir = Path('/home/amin/TSlib/enhanced_metrics/batch_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.summary_results = []
        self.failed_runs = []
        self.successful_runs = []
        
        # Performance tracking
        self.start_time = None
        self.total_runs = 0
        self.completed_runs = 0
        
    def run_batch_experiments(self, model_dataset_pairs: List[Tuple[str, str]], timeout: int = 120) -> Dict[str, Any]:
        """Run batch experiments with enhanced metrics collection"""
        
        self.start_time = time.time()
        self.total_runs = len(model_dataset_pairs)
        
        print(f"\n🚀 ENHANCED BATCH METRICS COLLECTION")
        print(f"{'='*70}")
        print(f"📊 Total Experiments: {self.total_runs}")
        print(f"⏰ Timeout per model: {timeout}s")
        print(f"📈 Enhanced Metrics: Time/Epoch, Peak GPU Memory, FLOPs/Epoch")
        print(f"🕐 Estimated Total Time: {(self.total_runs * timeout) / 60:.1f} minutes")
        print(f"{'='*70}")
        
        # Run each model-dataset combination
        for i, (model, dataset) in enumerate(model_dataset_pairs, 1):
            print(f"\n📋 EXPERIMENT {i}/{self.total_runs}")
            print(f"🏷️  Model: {model}")
            print(f"📊 Dataset: {dataset}")
            print(f"⏳ Progress: {((i-1)/self.total_runs)*100:.1f}% completed")
            
            try:
                # Run single model with enhanced metrics
                runner = EnhancedSingleModelRunner()
                results = runner.run_single_model(model, dataset, timeout)
                
                # Process results
                self._process_batch_results(model, dataset, results, i)
                self.successful_runs.append((model, dataset))
                
                print(f"✅ Experiment {i} completed successfully")
                
            except Exception as e:
                error_info = {
                    'model': model,
                    'dataset': dataset,
                    'experiment_number': i,
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.failed_runs.append(error_info)
                print(f"❌ Experiment {i} failed: {str(e)}")
                
            self.completed_runs = i
            
            # Progress update
            progress = (i / self.total_runs) * 100
            elapsed = time.time() - self.start_time
            estimated_total = elapsed / i * self.total_runs
            remaining = estimated_total - elapsed
            
            print(f"📊 Progress: {progress:.1f}% | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")
        
        # Generate comprehensive batch report
        batch_summary = self._generate_batch_summary()
        
        return batch_summary
    
    def _process_batch_results(self, model: str, dataset: str, results: Dict[str, Any], experiment_num: int):
        """Process individual experiment results for batch summary"""
        
        enhanced = results.get('enhanced_metrics', {})
        perf = results.get('performance_metrics', {})
        resources = results.get('resource_metrics', {})
        
        # Extract key metrics for summary
        summary_row = {
            'experiment_number': experiment_num,
            'model': model,
            'dataset': dataset,
            'model_family': enhanced.get('model_family', 'Unknown'),
            'dataset_type': enhanced.get('dataset_type', 'Unknown'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Core performance
            'accuracy': enhanced.get('accuracy', 0.0),
            'f1_score': perf.get('f1_score', 0.0),
            'auprc': perf.get('auprc', 0.0),
            'status': results.get('model_metrics', {}).get('status', 'unknown'),
            
            # ⭐ ENHANCED METRICS ⭐
            'time_per_epoch_seconds': enhanced.get('time_per_epoch_seconds', 0.0),
            'peak_gpu_memory_mb': enhanced.get('peak_gpu_memory_mb', 0.0),
            'peak_gpu_memory_gb': enhanced.get('peak_gpu_memory_gb', 0.0),
            'flops_per_epoch': enhanced.get('flops_per_epoch', 0.0),
            'total_gflops': enhanced.get('total_gflops', 0.0),
            
            # Training details
            'epochs_completed': enhanced.get('epochs_completed', 0),
            'total_training_time': enhanced.get('total_training_time', 0.0),
            'total_runtime': results.get('total_runtime', 0.0),
            
            # Efficiency metrics
            'accuracy_per_second': enhanced.get('accuracy_per_second', 0.0),
            'flops_efficiency': enhanced.get('flops_efficiency', 0.0),
            'memory_efficiency': enhanced.get('memory_efficiency', 0.0),
            'time_efficiency': enhanced.get('time_efficiency', 0.0),
            'gflops_per_second': enhanced.get('gflops_per_second', 0.0),
            
            # Resource utilization
            'avg_gpu_memory_mb': resources.get('avg_gpu_memory_mb', 0.0),
            'avg_gpu_utilization_percent': resources.get('avg_gpu_utilization_percent', 0.0),
            'avg_gpu_temperature_c': resources.get('avg_gpu_temperature_c', 0.0),
            'monitoring_samples_collected': resources.get('monitoring_samples_collected', 0),
            
            # Classifications
            'performance_class': enhanced.get('performance_class', 'unknown'),
            'runtime_class': enhanced.get('runtime_class', 'unknown'),
            'memory_class': enhanced.get('memory_class', 'unknown'),
            'computational_intensity': enhanced.get('computational_intensity', 'unknown'),
            
            # Energy estimation
            'estimated_energy_kwh': enhanced.get('estimated_energy_kwh', 0.0),
            'energy_efficiency': enhanced.get('energy_efficiency', 0.0),
        }
        
        self.summary_results.append(summary_row)
    
    def _generate_batch_summary(self) -> Dict[str, Any]:
        """Generate comprehensive batch summary with enhanced metrics analysis"""
        
        total_time = time.time() - self.start_time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Basic batch statistics
        batch_stats = {
            'batch_info': {
                'total_experiments': self.total_runs,
                'successful_runs': len(self.successful_runs),
                'failed_runs': len(self.failed_runs),
                'success_rate': len(self.successful_runs) / self.total_runs if self.total_runs > 0 else 0,
                'total_batch_time_seconds': total_time,
                'total_batch_time_minutes': total_time / 60,
                'avg_time_per_experiment': total_time / self.total_runs if self.total_runs > 0 else 0,
                'timestamp': timestamp,
            },
            'successful_experiments': self.successful_runs,
            'failed_experiments': self.failed_runs,
            'detailed_results': self.summary_results,
        }
        
        # Enhanced metrics analysis across all successful runs
        if self.summary_results:
            batch_stats['enhanced_analysis'] = self._analyze_enhanced_metrics_batch()
        
        # Save comprehensive batch results
        self._save_batch_results(batch_stats, timestamp)
        
        # Print batch summary
        self._print_batch_summary(batch_stats)
        
        return batch_stats
    
    def _analyze_enhanced_metrics_batch(self) -> Dict[str, Any]:
        """Analyze enhanced metrics across all successful batch runs"""
        
        successful_results = [r for r in self.summary_results if r['status'] == 'success']
        
        if not successful_results:
            return {'error': 'No successful runs to analyze'}
        
        # Extract enhanced metrics arrays
        accuracies = [r['accuracy'] for r in successful_results]
        time_per_epochs = [r['time_per_epoch_seconds'] for r in successful_results if r['time_per_epoch_seconds'] > 0]
        peak_memories = [r['peak_gpu_memory_mb'] for r in successful_results if r['peak_gpu_memory_mb'] > 0]
        flops_per_epochs = [r['flops_per_epoch'] for r in successful_results if r['flops_per_epoch'] > 0]
        
        # Calculate comprehensive statistics
        import numpy as np
        
        analysis = {
            'summary_stats': {
                'successful_experiments': len(successful_results),
                'experiments_with_time_per_epoch': len(time_per_epochs),
                'experiments_with_gpu_memory': len(peak_memories),
                'experiments_with_flops': len(flops_per_epochs),
            },
            
            # Accuracy analysis
            'accuracy_analysis': {
                'mean': np.mean(accuracies),
                'median': np.median(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'q25': np.percentile(accuracies, 25),
                'q75': np.percentile(accuracies, 75),
            } if accuracies else {},
            
            # ⭐ Time/Epoch Analysis ⭐
            'time_per_epoch_analysis': {
                'mean_seconds': np.mean(time_per_epochs),
                'median_seconds': np.median(time_per_epochs),
                'std_seconds': np.std(time_per_epochs),
                'min_seconds': np.min(time_per_epochs),
                'max_seconds': np.max(time_per_epochs),
                'q25_seconds': np.percentile(time_per_epochs, 25),
                'q75_seconds': np.percentile(time_per_epochs, 75),
                'coefficient_of_variation': np.std(time_per_epochs) / np.mean(time_per_epochs) if time_per_epochs and np.mean(time_per_epochs) > 0 else 0,
            } if time_per_epochs else {},
            
            # ⭐ Peak GPU Memory Analysis ⭐
            'peak_gpu_memory_analysis': {
                'mean_mb': np.mean(peak_memories),
                'median_mb': np.median(peak_memories),
                'std_mb': np.std(peak_memories),
                'min_mb': np.min(peak_memories),
                'max_mb': np.max(peak_memories),
                'q25_mb': np.percentile(peak_memories, 25),
                'q75_mb': np.percentile(peak_memories, 75),
                'mean_gb': np.mean(peak_memories) / 1024,
                'max_gb': np.max(peak_memories) / 1024,
            } if peak_memories else {},
            
            # ⭐ FLOPs/Epoch Analysis ⭐
            'flops_per_epoch_analysis': {
                'mean_flops': np.mean(flops_per_epochs),
                'median_flops': np.median(flops_per_epochs),
                'std_flops': np.std(flops_per_epochs),
                'min_flops': np.min(flops_per_epochs),
                'max_flops': np.max(flops_per_epochs),
                'q25_flops': np.percentile(flops_per_epochs, 25),
                'q75_flops': np.percentile(flops_per_epochs, 75),
                'mean_gflops': np.mean(flops_per_epochs) / 1e9,
                'max_gflops': np.max(flops_per_epochs) / 1e9,
            } if flops_per_epochs else {},
        }
        
        # Model family analysis
        model_families = {}
        for result in successful_results:
            family = result['model_family']
            if family not in model_families:
                model_families[family] = {
                    'count': 0,
                    'accuracies': [],
                    'time_per_epochs': [],
                    'peak_memories': [],
                    'flops_per_epochs': [],
                }
            model_families[family]['count'] += 1
            model_families[family]['accuracies'].append(result['accuracy'])
            if result['time_per_epoch_seconds'] > 0:
                model_families[family]['time_per_epochs'].append(result['time_per_epoch_seconds'])
            if result['peak_gpu_memory_mb'] > 0:
                model_families[family]['peak_memories'].append(result['peak_gpu_memory_mb'])
            if result['flops_per_epoch'] > 0:
                model_families[family]['flops_per_epochs'].append(result['flops_per_epoch'])
        
        # Calculate family statistics
        family_analysis = {}
        for family, data in model_families.items():
            family_stats = {
                'experiment_count': data['count'],
                'avg_accuracy': np.mean(data['accuracies']),
                'max_accuracy': np.max(data['accuracies']),
            }
            
            if data['time_per_epochs']:
                family_stats['avg_time_per_epoch'] = np.mean(data['time_per_epochs'])
                family_stats['min_time_per_epoch'] = np.min(data['time_per_epochs'])
                
            if data['peak_memories']:
                family_stats['avg_peak_memory_mb'] = np.mean(data['peak_memories'])
                family_stats['max_peak_memory_mb'] = np.max(data['peak_memories'])
                
            if data['flops_per_epochs']:
                family_stats['avg_flops_per_epoch'] = np.mean(data['flops_per_epochs'])
                family_stats['max_flops_per_epoch'] = np.max(data['flops_per_epochs'])
            
            family_analysis[family] = family_stats
        
        analysis['model_family_analysis'] = family_analysis
        
        # Performance champions
        if successful_results:
            # Accuracy champion
            accuracy_champion = max(successful_results, key=lambda x: x['accuracy'])
            
            # Efficiency champions
            flops_efficient = [r for r in successful_results if r['flops_efficiency'] > 0]
            memory_efficient = [r for r in successful_results if r['memory_efficiency'] > 0]
            time_efficient = [r for r in successful_results if r['time_efficiency'] > 0]
            
            champions = {
                'accuracy_champion': {
                    'model': accuracy_champion['model'],
                    'dataset': accuracy_champion['dataset'],
                    'accuracy': accuracy_champion['accuracy'],
                    'time_per_epoch': accuracy_champion['time_per_epoch_seconds'],
                    'peak_memory_mb': accuracy_champion['peak_gpu_memory_mb'],
                    'flops_per_epoch': accuracy_champion['flops_per_epoch'],
                }
            }
            
            if flops_efficient:
                flops_champion = max(flops_efficient, key=lambda x: x['flops_efficiency'])
                champions['flops_efficiency_champion'] = {
                    'model': flops_champion['model'],
                    'dataset': flops_champion['dataset'], 
                    'flops_efficiency': flops_champion['flops_efficiency'],
                    'accuracy': flops_champion['accuracy'],
                    'flops_per_epoch': flops_champion['flops_per_epoch'],
                }
            
            if memory_efficient:
                memory_champion = max(memory_efficient, key=lambda x: x['memory_efficiency'])
                champions['memory_efficiency_champion'] = {
                    'model': memory_champion['model'],
                    'dataset': memory_champion['dataset'],
                    'memory_efficiency': memory_champion['memory_efficiency'],
                    'accuracy': memory_champion['accuracy'],
                    'peak_memory_mb': memory_champion['peak_gpu_memory_mb'],
                }
            
            if time_efficient:
                time_champion = max(time_efficient, key=lambda x: x['time_efficiency'])
                champions['time_efficiency_champion'] = {
                    'model': time_champion['model'],
                    'dataset': time_champion['dataset'],
                    'time_efficiency': time_champion['time_efficiency'],
                    'accuracy': time_champion['accuracy'],
                    'time_per_epoch': time_champion['time_per_epoch_seconds'],
                }
            
            analysis['performance_champions'] = champions
        
        return analysis
    
    def _save_batch_results(self, batch_stats: Dict[str, Any], timestamp: str):
        """Save comprehensive batch results in multiple formats"""
        
        # Save detailed JSON results
        json_file = self.results_dir / f"enhanced_batch_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(batch_stats, f, indent=2)
        
        # Save CSV summary for easy analysis
        if self.summary_results:
            csv_file = self.results_dir / f"enhanced_batch_summary_{timestamp}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.summary_results[0].keys())
                writer.writeheader()
                writer.writerows(self.summary_results)
        
        # Save performance champions report
        if 'enhanced_analysis' in batch_stats and 'performance_champions' in batch_stats['enhanced_analysis']:
            champions_file = self.results_dir / f"performance_champions_{timestamp}.json"
            with open(champions_file, 'w') as f:
                json.dump(batch_stats['enhanced_analysis']['performance_champions'], f, indent=2)
        
        print(f"📁 Batch results saved:")
        print(f"   📊 Detailed: {json_file}")
        print(f"   📋 CSV Summary: {csv_file}")
    
    def _print_batch_summary(self, batch_stats: Dict[str, Any]):
        """Print comprehensive batch summary"""
        
        info = batch_stats['batch_info']
        analysis = batch_stats.get('enhanced_analysis', {})
        
        print(f"\n📊 ENHANCED BATCH METRICS SUMMARY")
        print(f"{'='*80}")
        print(f"🚀 Batch Execution Summary:")
        print(f"   Total Experiments: {info['total_experiments']}")
        print(f"   Successful: {info['successful_runs']} ({info['success_rate']*100:.1f}%)")
        print(f"   Failed: {info['failed_runs']}")
        print(f"   Total Time: {info['total_batch_time_minutes']:.1f} minutes")
        print(f"   Average Time/Experiment: {info['avg_time_per_experiment']:.1f} seconds")
        print()
        
        if 'summary_stats' in analysis:
            stats = analysis['summary_stats']
            print(f"📈 Enhanced Metrics Coverage:")
            print(f"   Experiments with Time/Epoch: {stats['experiments_with_time_per_epoch']}")
            print(f"   Experiments with GPU Memory: {stats['experiments_with_gpu_memory']}")
            print(f"   Experiments with FLOPs: {stats['experiments_with_flops']}")
            print()
        
        # Enhanced metrics statistics
        if 'time_per_epoch_analysis' in analysis and analysis['time_per_epoch_analysis']:
            time_stats = analysis['time_per_epoch_analysis']
            print(f"⏱️  Time/Epoch Statistics:")
            print(f"   Mean: {time_stats['mean_seconds']:.2f}s")
            print(f"   Median: {time_stats['median_seconds']:.2f}s")
            print(f"   Range: {time_stats['min_seconds']:.2f}s - {time_stats['max_seconds']:.2f}s")
            print(f"   Variability: {time_stats['coefficient_of_variation']:.2f}")
            print()
        
        if 'peak_gpu_memory_analysis' in analysis and analysis['peak_gpu_memory_analysis']:
            mem_stats = analysis['peak_gpu_memory_analysis']
            print(f"🔥 Peak GPU Memory Statistics:")
            print(f"   Mean: {mem_stats['mean_mb']:.0f}MB ({mem_stats['mean_gb']:.2f}GB)")
            print(f"   Median: {mem_stats['median_mb']:.0f}MB")
            print(f"   Range: {mem_stats['min_mb']:.0f}MB - {mem_stats['max_mb']:.0f}MB ({mem_stats['max_gb']:.2f}GB)")
            print()
        
        if 'flops_per_epoch_analysis' in analysis and analysis['flops_per_epoch_analysis']:
            flops_stats = analysis['flops_per_epoch_analysis']
            print(f"⚡ FLOPs/Epoch Statistics:")
            print(f"   Mean: {flops_stats['mean_flops']:.2e} ({flops_stats['mean_gflops']:.2f} GFLOPs)")
            print(f"   Median: {flops_stats['median_flops']:.2e}")
            print(f"   Range: {flops_stats['min_flops']:.2e} - {flops_stats['max_flops']:.2e} ({flops_stats['max_gflops']:.2f} GFLOPs)")
            print()
        
        # Performance champions
        if 'performance_champions' in analysis:
            champions = analysis['performance_champions']
            print(f"🏆 PERFORMANCE CHAMPIONS:")
            
            if 'accuracy_champion' in champions:
                acc_champ = champions['accuracy_champion']
                print(f"   🎯 Accuracy: {acc_champ['model']} on {acc_champ['dataset']} = {acc_champ['accuracy']:.4f}")
                
            if 'flops_efficiency_champion' in champions:
                flops_champ = champions['flops_efficiency_champion']
                print(f"   ⚡ FLOPs Efficiency: {flops_champ['model']} on {flops_champ['dataset']} = {flops_champ['flops_efficiency']:.6f} acc/GFLOP")
                
            if 'memory_efficiency_champion' in champions:
                mem_champ = champions['memory_efficiency_champion']
                print(f"   💾 Memory Efficiency: {mem_champ['model']} on {mem_champ['dataset']} = {mem_champ['memory_efficiency']:.4f} acc/GB")
                
            if 'time_efficiency_champion' in champions:
                time_champ = champions['time_efficiency_champion']
                print(f"   ⏱️  Time Efficiency: {time_champ['model']} on {time_champ['dataset']} = {time_champ['time_efficiency']:.6f} acc/s per epoch")
            print()
        
        # Model family performance
        if 'model_family_analysis' in analysis:
            print(f"🏷️  Model Family Performance:")
            family_analysis = analysis['model_family_analysis']
            for family, stats in family_analysis.items():
                print(f"   {family}: {stats['experiment_count']} experiments, avg accuracy {stats['avg_accuracy']:.4f}")
                if 'avg_time_per_epoch' in stats:
                    print(f"      Avg Time/Epoch: {stats['avg_time_per_epoch']:.2f}s")
                if 'avg_peak_memory_mb' in stats:
                    print(f"      Avg Peak Memory: {stats['avg_peak_memory_mb']:.0f}MB")
                if 'avg_flops_per_epoch' in stats:
                    print(f"      Avg FLOPs/Epoch: {stats['avg_flops_per_epoch']:.2e}")
            print()
        
        print(f"✅ Enhanced batch metrics collection completed!")
        print(f"{'='*80}")


def parse_config_file(config_path: str) -> List[Tuple[str, str]]:
    """Parse configuration file for model-dataset combinations"""
    
    pairs = []
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        models = config.get('models', [])
        datasets = config.get('datasets', [])
        
        # Generate all combinations
        for model in models:
            for dataset in datasets:
                pairs.append((model, dataset))
                
    except Exception as e:
        print(f"❌ Error parsing config file: {e}")
        return []
    
    return pairs


def main():
    """Main function for enhanced batch runner"""
    
    parser = argparse.ArgumentParser(description='Enhanced Batch Model Runner')
    parser.add_argument('--config', type=str, help='JSON config file with models and datasets')
    parser.add_argument('--models', type=str, help='Comma-separated list of models')
    parser.add_argument('--datasets', type=str, help='Comma-separated list of datasets')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per model in seconds')
    
    args = parser.parse_args()
    
    # Parse model-dataset combinations
    pairs = []
    
    if args.config:
        pairs = parse_config_file(args.config)
    elif args.models and args.datasets:
        models = [m.strip() for m in args.models.split(',')]
        datasets = [d.strip() for d in args.datasets.split(',')]
        pairs = [(model, dataset) for model in models for dataset in datasets]
    else:
        print("🎯 Enhanced Batch Model Runner")
        print("="*50)
        print("Usage:")
        print("  python enhanced_metrics/enhanced_batch_runner.py --config config.json [--timeout 120]")
        print("  python enhanced_metrics/enhanced_batch_runner.py --models TimesURL,BIOT --datasets Chinatown,AtrialFibrillation [--timeout 120]")
        print()
        print("Config file format (JSON):")
        print('  {')
        print('    "models": ["TimesURL", "BIOT", "CoST"],')
        print('    "datasets": ["Chinatown", "AtrialFibrillation"]')
        print('  }')
        print()
        print("⭐ Enhanced Metrics Collected:")
        print("  📅 Time/Epoch - Average training time per epoch")
        print("  🔥 Peak GPU Memory - Maximum GPU memory usage")
        print("  ⚡ FLOPs/Epoch - Computational complexity per epoch")
        print("  🚀 Comprehensive Efficiency Analysis")
        return
    
    if not pairs:
        print("❌ No valid model-dataset combinations found")
        return
    
    print(f"🚀 Starting enhanced batch collection with {len(pairs)} experiments:")
    for i, (model, dataset) in enumerate(pairs, 1):
        print(f"  {i:2d}. {model} on {dataset}")
    print()
    
    # Run batch experiments
    runner = EnhancedBatchRunner()
    batch_results = runner.run_batch_experiments(pairs, args.timeout)
    
    print("🎉 Enhanced batch metrics collection completed!")
    

if __name__ == "__main__":
    main()
