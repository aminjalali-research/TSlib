#!/usr/bin/env python3
"""
Custom Efficiency Optimizer for Your Specific TimeHUT Configuration
===================================================================

This script optimizes YOUR specific parameters:
- amc_instance=10.0, amc_temporal=7.53, amc_margin=0.3
- min_tau=0.05, max_tau=0.76, t_max=25
- epochs=200, batch_size=8

Author: Custom TimeHUT Optimizer
Date: August 28, 2025
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YourConfigOptimizer:
    """Optimize your specific TimeHUT configuration"""
    
    def __init__(self):
        self.results_dir = f"your_config_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # Your original parameters
        self.your_config = {
            'amc_instance': 10.0,
            'amc_temporal': 7.53,
            'amc_margin': 0.3,
            'min_tau': 0.05,
            'max_tau': 0.76,
            't_max': 25.0,
            'batch_size': 8,
            'epochs': 200,
            'temp_method': 'cosine_annealing'
        }
    
    def build_command(self, config):
        """Build training command with given config"""
        cmd = [
            '/home/amin/anaconda3/envs/tslib/bin/python',
            'train_unified_comprehensive.py',
            'Chinatown',
            f'opt_test_{int(time.time())}',
            '--loader', 'UCR',
            '--scenario', 'amc_temp',
            '--seed', '2002',
            '--amc-instance', str(config['amc_instance']),
            '--amc-temporal', str(config['amc_temporal']),
            '--amc-margin', str(config['amc_margin']),
            '--min-tau', str(config['min_tau']),
            '--max-tau', str(config['max_tau']),
            '--t-max', str(config['t_max']),
            '--batch-size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--temp-method', config['temp_method']
        ]
        return cmd
    
    def run_config(self, config, name):
        """Run training with given configuration"""
        logger.info(f"🧪 Testing: {name}")
        
        cmd = self.build_command(config)
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd='/home/amin/TSlib/models/timehut',
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )
            
            training_time = time.time() - start_time
            
            # Parse accuracy from output
            accuracy = 0.0
            for line in reversed(result.stdout.split('\\n')):
                if 'Final Accuracy:' in line:
                    try:
                        accuracy = float(line.split('Final Accuracy:')[1].strip())
                        break
                    except (ValueError, IndexError):
                        continue
            
            return {
                'name': name,
                'accuracy': accuracy,
                'training_time': training_time,
                'status': 'success' if result.returncode == 0 else 'failed',
                'config': config
            }
            
        except subprocess.TimeoutExpired:
            return {'name': name, 'status': 'timeout', 'training_time': 600}
        except Exception as e:
            return {'name': name, 'status': 'error', 'error': str(e)}
    
    def optimize_your_config(self):
        """Optimize your specific configuration"""
        logger.info("🚀 Optimizing YOUR specific TimeHUT configuration...")
        
        results = []
        
        # 1. Baseline (your original config)
        baseline_result = self.run_config(self.your_config, "Your Original Config")
        results.append(baseline_result)
        
        if baseline_result['status'] != 'success':
            logger.error("❌ Original configuration failed - cannot optimize")
            return results
        
        baseline_acc = baseline_result['accuracy']
        baseline_time = baseline_result['training_time']
        
        logger.info(f"✅ Baseline: {baseline_acc:.4f} accuracy in {baseline_time:.1f}s")
        
        # 2. Reduced epochs (early stopping simulation)
        for epoch_reduction in [0.8, 0.6, 0.4]:  # 20%, 40%, 60% reduction
            optimized_epochs = int(self.your_config['epochs'] * epoch_reduction)
            
            config = self.your_config.copy()
            config['epochs'] = optimized_epochs
            
            result = self.run_config(config, f"Reduced Epochs ({optimized_epochs})")
            results.append(result)
            
            if result['status'] == 'success':
                acc_drop = baseline_acc - result['accuracy']
                time_saved = (baseline_time - result['training_time']) / baseline_time * 100
                
                logger.info(f"  📊 {optimized_epochs} epochs: {result['accuracy']:.4f} (-{acc_drop:.4f}), {time_saved:.1f}% faster")
        
        # 3. Increased batch size
        for batch_mult in [2, 4]:  # 16, 32 batch size
            optimized_batch = self.your_config['batch_size'] * batch_mult
            
            config = self.your_config.copy()
            config['batch_size'] = optimized_batch
            config['epochs'] = int(self.your_config['epochs'] * 0.7)  # Reduce epochs for faster testing
            
            result = self.run_config(config, f"Batch Size {optimized_batch}")
            results.append(result)
            
            if result['status'] == 'success':
                logger.info(f"  📊 Batch {optimized_batch}: {result['accuracy']:.4f} accuracy")
        
        # 4. Alternative schedulers
        alternative_schedulers = [
            ('polynomial_decay', {'temp_power': '2.5'}),
            ('exponential_decay', {'temp_decay_rate': '0.95'}),
            ('linear_decay', {})
        ]
        
        for scheduler, extra_args in alternative_schedulers:
            config = self.your_config.copy()
            config['temp_method'] = scheduler
            config['epochs'] = int(self.your_config['epochs'] * 0.6)  # Quick test
            
            # Build command with extra scheduler args
            cmd = self.build_command(config)
            for arg_name, arg_value in extra_args.items():
                cmd.extend([f'--{arg_name.replace("_", "-")}', arg_value])
            
            logger.info(f"🧪 Testing scheduler: {scheduler}")
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd='/home/amin/TSlib/models/timehut',
                    capture_output=True,
                    text=True,
                    timeout=400
                )
                
                training_time = time.time() - start_time
                
                # Parse accuracy
                accuracy = 0.0
                for line in reversed(result.stdout.split('\\n')):
                    if 'Final Accuracy:' in line:
                        try:
                            accuracy = float(line.split('Final Accuracy:')[1].strip())
                            break
                        except (ValueError, IndexError):
                            continue
                
                scheduler_result = {
                    'name': f"Scheduler {scheduler}",
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'status': 'success' if result.returncode == 0 else 'failed',
                    'config': config
                }
                
                results.append(scheduler_result)
                
                if scheduler_result['status'] == 'success':
                    logger.info(f"  📊 {scheduler}: {accuracy:.4f} accuracy")
                    
            except Exception as e:
                logger.error(f"  ❌ {scheduler} failed: {e}")
        
        # 5. Combined best optimizations
        logger.info("🔥 Testing combined optimizations...")
        
        # Find best individual optimizations
        successful_results = [r for r in results if r['status'] == 'success']
        
        # Combined config: reduced epochs + larger batch + best scheduler
        combined_config = self.your_config.copy()
        combined_config['epochs'] = int(self.your_config['epochs'] * 0.6)  # 40% reduction
        combined_config['batch_size'] = 16  # Double batch size
        combined_config['temp_method'] = 'polynomial_decay'
        
        # Build command with polynomial decay power
        cmd = self.build_command(combined_config)
        cmd.extend(['--temp-power', '2.5'])
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd='/home/amin/TSlib/models/timehut',
                capture_output=True,
                text=True,
                timeout=600
            )
            
            training_time = time.time() - start_time
            
            # Parse accuracy
            accuracy = 0.0
            for line in reversed(result.stdout.split('\\n')):
                if 'Final Accuracy:' in line:
                    try:
                        accuracy = float(line.split('Final Accuracy:')[1].strip())
                        break
                    except (ValueError, IndexError):
                        continue
            
            combined_result = {
                'name': 'Combined Optimizations',
                'accuracy': accuracy,
                'training_time': training_time,
                'status': 'success' if result.returncode == 0 else 'failed',
                'config': combined_config,
                'optimizations': ['reduced_epochs_60%', 'batch_size_16', 'polynomial_decay']
            }
            
            results.append(combined_result)
            
            if combined_result['status'] == 'success':
                time_reduction = (baseline_time - training_time) / baseline_time * 100
                accuracy_change = accuracy - baseline_acc
                
                logger.info(f"✅ Combined: {accuracy:.4f} ({accuracy_change:+.4f}), {time_reduction:.1f}% faster")
            
        except Exception as e:
            logger.error(f"❌ Combined optimization failed: {e}")
        
        return results
    
    def generate_report(self, results):
        """Generate optimization report"""
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            logger.error("❌ No successful optimization results")
            return
        
        baseline = successful_results[0]  # Your original config
        
        report = []
        report.append("# Your TimeHUT Configuration Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Original configuration
        report.append("## 🎯 Your Original Configuration")
        report.append(f"- **AMC Instance**: {self.your_config['amc_instance']}")
        report.append(f"- **AMC Temporal**: {self.your_config['amc_temporal']}")
        report.append(f"- **AMC Margin**: {self.your_config['amc_margin']}")
        report.append(f"- **Temperature Range**: {self.your_config['min_tau']} - {self.your_config['max_tau']}")
        report.append(f"- **T-Max**: {self.your_config['t_max']}")
        report.append(f"- **Batch Size**: {self.your_config['batch_size']}")
        report.append(f"- **Epochs**: {self.your_config['epochs']}")
        report.append(f"- **Scheduler**: {self.your_config['temp_method']}")
        report.append("")
        
        # Baseline performance
        report.append("## 📊 Original Performance")
        report.append(f"- **Accuracy**: {baseline['accuracy']:.4f}")
        report.append(f"- **Training Time**: {baseline['training_time']:.1f} seconds")
        report.append("")
        
        # Optimization results
        report.append("## ⚡ Optimization Results")
        
        for result in successful_results[1:]:  # Skip baseline
            if 'Combined' in result['name']:
                report.append(f"### 🚀 {result['name']}")
                report.append(f"- **Final Accuracy**: {result['accuracy']:.4f}")
                time_reduction = (baseline['training_time'] - result['training_time']) / baseline['training_time'] * 100
                accuracy_change = result['accuracy'] - baseline['accuracy']
                report.append(f"- **Training Time**: {result['training_time']:.1f}s ({time_reduction:.1f}% faster)")
                report.append(f"- **Accuracy Change**: {accuracy_change:+.4f}")
                
                if 'optimizations' in result:
                    report.append(f"- **Optimizations**: {', '.join(result['optimizations'])}")
                
                # Generate optimized command
                config = result['config']
                cmd_parts = [
                    'python train_unified_comprehensive.py Chinatown your_optimized',
                    '--loader UCR --scenario amc_temp',
                    f"--amc-instance {config['amc_instance']}",
                    f"--amc-temporal {config['amc_temporal']}",
                    f"--amc-margin {config['amc_margin']}",
                    f"--min-tau {config['min_tau']}",
                    f"--max-tau {config['max_tau']}",
                    f"--t-max {config['t_max']}",
                    f"--batch-size {config['batch_size']}",
                    f"--epochs {config['epochs']}",
                    f"--temp-method {config['temp_method']}",
                    '--temp-power 2.5 --verbose'
                ]
                
                report.append("")
                report.append("**Optimized Command:**")
                report.append("```bash")
                report.append(" ".join(cmd_parts))
                report.append("```")
                report.append("")
        
        # Recommendations
        best_result = max(successful_results, key=lambda x: x['accuracy'] if x['status'] == 'success' else 0)
        fastest_result = min([r for r in successful_results if r['status'] == 'success'], 
                           key=lambda x: x['training_time'])
        
        report.append("## 🎯 Recommendations")
        
        if best_result != baseline:
            report.append(f"✅ **Best Accuracy**: {best_result['name']} achieved {best_result['accuracy']:.4f}")
        
        if fastest_result != baseline:
            time_saved = (baseline['training_time'] - fastest_result['training_time']) / baseline['training_time'] * 100
            report.append(f"⚡ **Fastest Training**: {fastest_result['name']} saved {time_saved:.1f}% time")
        
        report.append("")
        
        report_text = "\\n".join(report)
        
        # Save report
        report_file = Path(self.results_dir) / "your_config_optimization_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"📄 Your optimization report saved to: {report_file}")
        return report_text

def main():
    """Main execution"""
    optimizer = YourConfigOptimizer()
    
    try:
        logger.info("🎯 Starting optimization of YOUR specific TimeHUT configuration...")
        
        results = optimizer.optimize_your_config()
        
        # Generate report
        optimizer.generate_report(results)
        
        logger.info("✅ Your configuration optimization completed!")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise

if __name__ == '__main__':
    main()
