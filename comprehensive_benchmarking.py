#!/usr/bin/env python3
"""
TSlib Comprehensive Benchmarking Script
========================================

Run all working models and collect comprehensive metrics for comparison.
Based on working models from TaskGuide.md and existing results.
"""

import subprocess
import json
import time
import os
from datetime import datetime
from pathlib import Path

class ComprehensiveBenchmarking:
    def __init__(self):
        self.working_models = {
            # TS2vec Collection (tslib environment) - 100% success rate
            'TS2vec_family': {
                'models': ['TS2vec', 'TimeHUT', 'SoftCLT'],
                'env': 'tslib',
                'datasets': ['Chinatown', 'AtrialFibrillation'],
                'timeout': 120,
                'success_rate': '100%'
            },
            
            # VQ-MTM Collection (vq_mtm environment) - 89% success rate  
            'VQ-MTM_family': {
                'models': ['BIOT', 'VQ_MTM', 'DCRNN', 'Ti_MAE', 'SimMTM'],
                'env': 'vq_mtm',
                'datasets': ['AtrialFibrillation'],  # Multivariate specialist
                'timeout': 180,
                'success_rate': '89%'
            },
            
            # MF-CLR Collection (mfclr environment) - Enhanced metrics
            'MF-CLR_family': {
                'models': ['TNC', 'CPC', 'CoST', 'TS_TCC', 'TLoss', 'TFC', 'MF_CLR'],
                'env': 'mfclr', 
                'datasets': ['Chinatown', 'AtrialFibrillation'],
                'timeout': 400,  # CoST is slow but has enhanced GPU/FLOPs metrics
                'success_rate': '85%'
            },
            
            # TimesURL (timesurl environment) - Champion performer
            'TimesURL_family': {
                'models': ['TimesURL'],
                'env': 'timesurl',
                'datasets': ['Chinatown', 'AtrialFibrillation'], 
                'timeout': 120,
                'success_rate': '100%'
            }
        }
        
        self.results_dir = Path('/home/amin/TSlib/results/comprehensive_benchmarking')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this benchmarking session
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def run_comprehensive_benchmarking(self):
        """Run comprehensive benchmarking across all working model collections"""
        
        print("🚀 TSlib Comprehensive Benchmarking")
        print("=" * 60)
        print(f"📅 Session: {self.timestamp}")
        print(f"📁 Results: {self.results_dir}")
        print(f"🎯 Target: All working models with comprehensive metrics")
        print()
        
        all_results = {}
        total_experiments = 0
        successful_experiments = 0
        
        for family_name, family_config in self.working_models.items():
            print(f"🔥 Starting {family_name} Collection")
            print(f"   Models: {', '.join(family_config['models'])}")
            print(f"   Environment: {family_config['env']}")
            print(f"   Success Rate: {family_config['success_rate']}")
            print()
            
            family_results = self.benchmark_model_family(family_name, family_config)
            all_results[family_name] = family_results
            
            # Count experiments
            for dataset_results in family_results.values():
                for model_results in dataset_results.values():
                    total_experiments += 1
                    if model_results.get('status') == 'success':
                        successful_experiments += 1
            
            print(f"✅ Completed {family_name}")
            print("-" * 40)
            print()
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_results, total_experiments, successful_experiments)
        
        return all_results
    
    def benchmark_model_family(self, family_name, config):
        """Benchmark all models in a family across all datasets"""
        
        family_results = {}
        
        for dataset in config['datasets']:
            print(f"  📊 Testing on {dataset} dataset...")
            dataset_results = {}
            
            for model in config['models']:
                print(f"    🔬 Running {model}...")
                
                result = self.run_single_benchmark(
                    model=model,
                    dataset=dataset,
                    env=config['env'],
                    timeout=config['timeout']
                )
                
                dataset_results[model] = result
                
                # Quick status report
                if result['status'] == 'success':
                    accuracy = result.get('accuracy', 'N/A')
                    runtime = result.get('training_time', 'N/A')
                    print(f"      ✅ Success: {accuracy} accuracy in {runtime}")
                else:
                    print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
            
            family_results[dataset] = dataset_results
        
        return family_results
    
    def run_single_benchmark(self, model, dataset, env, timeout):
        """Run single model benchmark with comprehensive metrics collection"""
        
        start_time = time.time()
        
        # Build command
        cmd = [
            'python', 'unified/master_benchmark_pipeline.py',
            '--models', model,
            '--datasets', dataset,
            '--optimization',
            '--optimization-mode', 'fair',  # Fair comparison with seed=42
            '--timeout', str(timeout)
        ]
        
        try:
            # Set environment variables for proper execution
            env_vars = os.environ.copy()
            env_vars['MKL_THREADING_LAYER'] = 'GNU'  # Fix MKL conflicts
            env_vars['CONDA_DEFAULT_ENV'] = env
            
            # Run benchmark
            result = subprocess.run(
                cmd,
                cwd='/home/amin/TSlib',
                capture_output=True,
                text=True,
                timeout=timeout + 30,  # Add buffer for pipeline overhead
                env=env_vars
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                metrics = self.parse_benchmark_output(result.stdout, result.stderr)
                metrics.update({
                    'status': 'success',
                    'execution_time': f"{execution_time:.1f}s",
                    'command': ' '.join(cmd),
                    'environment': env
                })
                return metrics
            else:
                return {
                    'status': 'failed',
                    'error': result.stderr[-500:],  # Last 500 chars of error
                    'execution_time': f"{execution_time:.1f}s",
                    'command': ' '.join(cmd),
                    'environment': env
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'error': f'Exceeded {timeout}s timeout',
                'execution_time': f"{timeout}s+",
                'command': ' '.join(cmd),
                'environment': env
            }
        except Exception as e:
            return {
                'status': 'exception',
                'error': str(e),
                'execution_time': f"{time.time() - start_time:.1f}s",
                'command': ' '.join(cmd),
                'environment': env
            }
    
    def parse_benchmark_output(self, stdout, stderr):
        """Parse benchmark output to extract comprehensive metrics"""
        
        metrics = {}
        
        # Look for accuracy in various formats
        import re
        
        # Standard accuracy patterns
        accuracy_patterns = [
            r'Accuracy[:\s]+([0-9.]+)',
            r'Final accuracy[:\s]+([0-9.]+)',
            r'Test accuracy[:\s]+([0-9.]+)',
            r'Classification accuracy[:\s]+([0-9.]+)',
            r'acc[:\s]+([0-9.]+)',
        ]
        
        for pattern in accuracy_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['accuracy'] = f"{float(match.group(1)):.4f}"
                break
        
        # F1-Score patterns
        f1_patterns = [
            r'F1[:\s]+([0-9.]+)',
            r'F1-score[:\s]+([0-9.]+)',
            r'f1_score[:\s]+([0-9.]+)',
        ]
        
        for pattern in f1_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['f1_score'] = f"{float(match.group(1)):.4f}"
                break
        
        # Training time patterns
        time_patterns = [
            r'Training completed in ([0-9.]+)s',
            r'Total training time[:\s]+([0-9.]+)s',
            r'Completed in ([0-9.]+)s',
            r'Runtime[:\s]+([0-9.]+)s',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['training_time'] = f"{float(match.group(1)):.1f}s"
                break
        
        # GPU Memory patterns (from MF-CLR enhanced metrics)
        gpu_patterns = [
            r'GPU Memory[:\s]+([0-9.]+)\s*MB',
            r'Peak GPU[:\s]+([0-9.]+)\s*MB',
            r'Memory usage[:\s]+([0-9.]+)\s*MB',
        ]
        
        for pattern in gpu_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['gpu_memory'] = f"{int(float(match.group(1)))} MB"
                break
        
        # FLOPs patterns (from MF-CLR enhanced metrics)
        flops_patterns = [
            r'FLOPs[:\s]+([0-9.]+[KMGTB]?)',
            r'FLOP count[:\s]+([0-9.]+[KMGTB]?)',
            r'Floating point operations[:\s]+([0-9.]+[KMGTB]?)',
        ]
        
        for pattern in flops_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['flops'] = match.group(1)
                break
        
        # Temperature patterns (from MF-CLR enhanced metrics)
        temp_patterns = [
            r'Temperature[:\s]+([0-9.]+)°?C',
            r'GPU Temp[:\s]+([0-9.]+)°?C',
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['gpu_temperature'] = f"{int(float(match.group(1)))}°C"
                break
        
        # AUPRC patterns
        auprc_patterns = [
            r'AUPRC[:\s]+([0-9.]+)',
            r'AUC-PR[:\s]+([0-9.]+)',
            r'Area under PR curve[:\s]+([0-9.]+)',
        ]
        
        for pattern in auprc_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['auprc'] = f"{float(match.group(1)):.4f}"
                break
        
        # Precision/Recall patterns
        precision_patterns = [r'Precision[:\s]+([0-9.]+)']
        recall_patterns = [r'Recall[:\s]+([0-9.]+)']
        
        for pattern in precision_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['precision'] = f"{float(match.group(1)):.4f}"
                break
        
        for pattern in recall_patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                metrics['recall'] = f"{float(match.group(1)):.4f}"
                break
        
        # Epochs completed
        epoch_patterns = [
            r'Epoch ([0-9]+)/([0-9]+)',
            r'([0-9]+) epochs completed',
            r'Training for ([0-9]+) epochs',
        ]
        
        for pattern in epoch_patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                if len(matches[-1]) == 2:  # Epoch x/y format
                    metrics['epochs'] = matches[-1][1]
                else:  # Single number format
                    metrics['epochs'] = matches[-1]
                break
        
        return metrics
    
    def generate_comprehensive_report(self, all_results, total_experiments, successful_experiments):
        """Generate comprehensive benchmarking report"""
        
        success_rate = (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0
        
        # Save JSON results
        json_file = self.results_dir / f'comprehensive_benchmarking_{self.timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate markdown report
        report_file = self.results_dir / f'comprehensive_report_{self.timestamp}.md'
        
        with open(report_file, 'w') as f:
            f.write(f"# TSlib Comprehensive Benchmarking Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}\n")
            f.write(f"**Session ID**: {self.timestamp}\n")
            f.write(f"**Total Experiments**: {total_experiments}\n")
            f.write(f"**Successful**: {successful_experiments}\n")
            f.write(f"**Success Rate**: {success_rate:.1f}%\n\n")
            
            # Performance champions
            f.write("## 🏆 Performance Champions\n\n")
            champions = self.identify_champions(all_results)
            for category, champion in champions.items():
                f.write(f"- **{category}**: {champion}\n")
            f.write("\n")
            
            # Model collection results
            f.write("## 📊 Results by Model Collection\n\n")
            
            for family_name, family_results in all_results.items():
                f.write(f"### {family_name}\n\n")
                f.write("| Model | Dataset | Accuracy | F1-Score | Training Time | GPU Memory | FLOPs | Status |\n")
                f.write("|-------|---------|----------|----------|---------------|------------|-------|---------|\n")
                
                for dataset, models in family_results.items():
                    for model, result in models.items():
                        status_icon = "✅" if result['status'] == 'success' else "❌"
                        accuracy = result.get('accuracy', 'N/A')
                        f1_score = result.get('f1_score', 'N/A')
                        training_time = result.get('training_time', 'N/A')
                        gpu_memory = result.get('gpu_memory', 'N/A')
                        flops = result.get('flops', 'N/A')
                        
                        f.write(f"| {model} | {dataset} | {accuracy} | {f1_score} | {training_time} | {gpu_memory} | {flops} | {status_icon} {result['status'].title()} |\n")
                f.write("\n")
            
            # Computational analysis
            f.write("## 💻 Computational Analysis\n\n")
            computational_stats = self.analyze_computational_performance(all_results)
            for stat_name, stat_value in computational_stats.items():
                f.write(f"- **{stat_name}**: {stat_value}\n")
            f.write("\n")
            
            # Dataset specialization
            f.write("## 🎯 Dataset Specialization Analysis\n\n")
            specialization = self.analyze_dataset_specialization(all_results)
            for dataset, best_models in specialization.items():
                f.write(f"### {dataset} Dataset Champions\n")
                for i, (model, accuracy) in enumerate(best_models[:3], 1):
                    medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else ""
                    f.write(f"{i}. **{model}** - {accuracy} {medal}\n")
                f.write("\n")
        
        print("📊 COMPREHENSIVE BENCHMARKING COMPLETE")
        print("=" * 60)
        print(f"📁 Results saved to: {report_file}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"🎯 Total experiments: {total_experiments}")
        print(f"✅ Successful: {successful_experiments}")
        print()
        print(f"📋 Next steps:")
        print(f"   1. Review report: {report_file}")
        print(f"   2. Check JSON data: {json_file}")
        print(f"   3. Analyze performance champions and patterns")
        
    def identify_champions(self, all_results):
        """Identify performance champions across categories"""
        
        champions = {}
        best_accuracy = 0
        best_speed = float('inf')
        best_efficiency = 0
        
        for family_results in all_results.values():
            for dataset_results in family_results.values():
                for model, result in dataset_results.items():
                    if result['status'] != 'success':
                        continue
                    
                    # Accuracy champion
                    accuracy = float(result.get('accuracy', 0))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        champions['Highest Accuracy'] = f"{model} ({accuracy:.1%})"
                    
                    # Speed champion (lowest training time)
                    if 'training_time' in result:
                        time_str = result['training_time'].replace('s', '')
                        try:
                            training_time = float(time_str)
                            if training_time < best_speed:
                                best_speed = training_time
                                champions['Fastest Training'] = f"{model} ({training_time:.1f}s)"
                        except:
                            pass
        
        return champions
    
    def analyze_computational_performance(self, all_results):
        """Analyze computational performance across all models"""
        
        stats = {}
        total_models = 0
        models_with_gpu_data = 0
        models_with_flops_data = 0
        
        for family_results in all_results.values():
            for dataset_results in family_results.values():
                for model, result in dataset_results.items():
                    if result['status'] == 'success':
                        total_models += 1
                        if 'gpu_memory' in result:
                            models_with_gpu_data += 1
                        if 'flops' in result:
                            models_with_flops_data += 1
        
        stats['Total Working Models'] = str(total_models)
        stats['Models with GPU Metrics'] = f"{models_with_gpu_data}/{total_models} ({models_with_gpu_data/total_models*100:.0f}%)"
        stats['Models with FLOPs Data'] = f"{models_with_flops_data}/{total_models} ({models_with_flops_data/total_models*100:.0f}%)"
        
        return stats
    
    def analyze_dataset_specialization(self, all_results):
        """Analyze which models perform best on each dataset"""
        
        dataset_performance = {}
        
        for family_results in all_results.values():
            for dataset, models in family_results.items():
                if dataset not in dataset_performance:
                    dataset_performance[dataset] = []
                
                for model, result in models.items():
                    if result['status'] == 'success' and 'accuracy' in result:
                        accuracy = float(result['accuracy'])
                        dataset_performance[dataset].append((model, accuracy))
        
        # Sort by accuracy for each dataset
        for dataset in dataset_performance:
            dataset_performance[dataset].sort(key=lambda x: x[1], reverse=True)
            # Format accuracy as percentage
            dataset_performance[dataset] = [(model, f"{acc:.1%}") for model, acc in dataset_performance[dataset]]
        
        return dataset_performance


def main():
    """Main execution function"""
    
    print("🚀 TSlib Comprehensive Benchmarking")
    print("=" * 60)
    print()
    print("This script will run all working models and collect comprehensive metrics:")
    print("• Accuracy, F1-Score, AUPRC")
    print("• Training time and computational efficiency") 
    print("• GPU memory usage and temperature")
    print("• FLOPs counting for algorithm complexity")
    print("• Cross-dataset performance comparison")
    print()
    print("🚀 Starting automatic comprehensive benchmarking...")
    print()
    
    # Initialize and run benchmarking
    benchmarker = ComprehensiveBenchmarking()
    results = benchmarker.run_comprehensive_benchmarking()
    
    print()
    print("🎉 Comprehensive benchmarking session complete!")
    print("Check the results directory for detailed reports and analysis.")


if __name__ == "__main__":
    main()
