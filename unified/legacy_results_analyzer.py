#!/usr/bin/env python3
"""
Benchmark Results Analysis and Visualization
Aggregates results from multiple benchmark runs and creates comparison charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path
import numpy as np

class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, results_dir='/home/amin/TimesURL/benchmark_results'):
        self.results_dir = Path(results_dir)
        self.df = None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_all_results(self, pattern="*_benchmark_*.csv"):
        """Load all benchmark CSV files into a single DataFrame"""
        csv_files = list(self.results_dir.glob(f"**/{pattern}"))
        
        if not csv_files:
            print(f"❌ No benchmark files found in {self.results_dir}")
            return None
            
        print(f"📁 Found {len(csv_files)} benchmark result files")
        
        all_dfs = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                df['result_file'] = file_path.name
                all_dfs.append(df)
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
                
        if all_dfs:
            self.df = pd.concat(all_dfs, ignore_index=True)
            print(f"✅ Loaded {len(self.df)} benchmark records")
            return self.df
        else:
            print("❌ No valid benchmark data found")
            return None
            
    def create_comparison_plots(self, output_dir=None):
        """Create comprehensive comparison visualizations"""
        if self.df is None:
            print("❌ No data loaded. Call load_all_results() first.")
            return
            
        if output_dir is None:
            output_dir = self.results_dir / 'analysis_plots'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Filter successful runs only
        successful_df = self.df[self.df['success'] == True].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful benchmark runs found")
            return
            
        print(f"📊 Creating plots from {len(successful_df)} successful runs...")
        
        # 1. Accuracy Comparison
        plt.figure(figsize=(12, 8))
        
        # Box plot for accuracy by method
        plt.subplot(2, 2, 1)
        successful_df.boxplot(column='accuracy', by='method', ax=plt.gca())
        plt.title('Accuracy by Method')
        plt.xlabel('Method')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # 2. Training Time Comparison  
        plt.subplot(2, 2, 2)
        successful_df.boxplot(column='total_training_time', by='method', ax=plt.gca())
        plt.title('Total Training Time by Method')
        plt.xlabel('Method')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(rotation=45)
        
        # 3. GPU Memory Usage
        plt.subplot(2, 2, 3)
        successful_df.boxplot(column='peak_gpu_memory_gb', by='method', ax=plt.gca())
        plt.title('Peak GPU Memory by Method')
        plt.xlabel('Method')
        plt.ylabel('Peak Memory (GB)')
        plt.xticks(rotation=45)
        
        # 4. GPU Utilization
        plt.subplot(2, 2, 4)
        successful_df.boxplot(column='avg_gpu_utilization', by='method', ax=plt.gca())
        plt.title('Average GPU Utilization by Method')
        plt.xlabel('Method')
        plt.ylabel('GPU Utilization (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'method_comparison_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed comparison by dataset
        if len(successful_df['dataset'].unique()) > 1:
            # Accuracy vs Training Time scatter plot
            plt.figure(figsize=(12, 8))
            
            for method in successful_df['method'].unique():
                method_data = successful_df[successful_df['method'] == method]
                plt.scatter(method_data['total_training_time'], method_data['accuracy'], 
                           label=method.upper(), s=60, alpha=0.7)
                
            plt.xlabel('Total Training Time (seconds)')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Training Time by Method')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'accuracy_vs_time_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Heatmap of performance by dataset and method
            pivot_accuracy = successful_df.pivot_table(values='accuracy', index='dataset', 
                                                     columns='method', aggfunc='mean')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Accuracy'})
            plt.title('Average Accuracy by Dataset and Method')
            plt.tight_layout()
            plt.savefig(output_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Plots saved to {output_dir}")
        
    def generate_summary_report(self, output_file=None):
        """Generate a comprehensive text summary report"""
        if self.df is None:
            print("❌ No data loaded. Call load_all_results() first.")
            return
            
        if output_file is None:
            output_file = self.results_dir / 'benchmark_summary_report.txt'
        else:
            output_file = Path(output_file)
            
        successful_df = self.df[self.df['success'] == True].copy()
        
        with open(output_file, 'w') as f:
            f.write("TIME SERIES METHODS COMPREHENSIVE BENCHMARK REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Benchmark Runs: {len(self.df)}\n")
            f.write(f"Successful Runs: {len(successful_df)}\n")
            f.write(f"Failed Runs: {len(self.df) - len(successful_df)}\n\n")
            
            f.write("DATASETS TESTED:\n")
            f.write("-" * 20 + "\n")
            for dataset in sorted(self.df['dataset'].unique()):
                count = len(self.df[self.df['dataset'] == dataset])
                f.write(f"• {dataset}: {count} runs\n")
            f.write("\n")
            
            f.write("METHODS COMPARED:\n")
            f.write("-" * 20 + "\n")
            for method in sorted(self.df['method'].unique()):
                count = len(self.df[self.df['method'] == method])
                success_count = len(successful_df[successful_df['method'] == method])
                f.write(f"• {method.upper()}: {success_count}/{count} successful\n")
            f.write("\n")
            
            if len(successful_df) > 0:
                f.write("PERFORMANCE SUMMARY (SUCCESSFUL RUNS ONLY):\n")
                f.write("-" * 45 + "\n\n")
                
                # Overall statistics by method
                for method in sorted(successful_df['method'].unique()):
                    method_data = successful_df[successful_df['method'] == method]
                    
                    f.write(f"{method.upper()} PERFORMANCE:\n")
                    f.write(f"  Runs: {len(method_data)}\n")
                    f.write(f"  Accuracy: {method_data['accuracy'].mean():.4f} ± {method_data['accuracy'].std():.4f}\n")
                    f.write(f"  F1 Score: {method_data['f1_score'].mean():.4f} ± {method_data['f1_score'].std():.4f}\n")
                    f.write(f"  Avg Training Time: {method_data['total_training_time'].mean():.2f}s ± {method_data['total_training_time'].std():.2f}s\n")
                    f.write(f"  Avg Time/Epoch: {method_data['time_per_epoch'].mean():.2f}s ± {method_data['time_per_epoch'].std():.2f}s\n")
                    f.write(f"  Peak GPU Memory: {method_data['peak_gpu_memory_gb'].mean():.2f}GB ± {method_data['peak_gpu_memory_gb'].std():.2f}GB\n")
                    f.write(f"  Avg GPU Utilization: {method_data['avg_gpu_utilization'].mean():.1f}% ± {method_data['avg_gpu_utilization'].std():.1f}%\n\n")
                
                # Best performing method by metric
                f.write("BEST PERFORMING METHOD BY METRIC:\n")
                f.write("-" * 35 + "\n")
                
                method_avg = successful_df.groupby('method').mean()
                
                best_acc = method_avg['accuracy'].idxmax()
                best_f1 = method_avg['f1_score'].idxmax()
                fastest = method_avg['total_training_time'].idxmin()
                most_efficient = method_avg['peak_gpu_memory_gb'].idxmin()
                
                f.write(f"• Highest Accuracy: {best_acc.upper()} ({method_avg.loc[best_acc, 'accuracy']:.4f})\n")
                f.write(f"• Highest F1 Score: {best_f1.upper()} ({method_avg.loc[best_f1, 'f1_score']:.4f})\n")
                f.write(f"• Fastest Training: {fastest.upper()} ({method_avg.loc[fastest, 'total_training_time']:.2f}s avg)\n")
                f.write(f"• Most Memory Efficient: {most_efficient.upper()} ({method_avg.loc[most_efficient, 'peak_gpu_memory_gb']:.2f}GB avg)\n\n")
                
        print(f"📄 Summary report saved to {output_file}")
        
    def create_performance_ranking(self):
        """Create a performance ranking across all metrics"""
        if self.df is None:
            print("❌ No data loaded. Call load_all_results() first.")
            return None
            
        successful_df = self.df[self.df['success'] == True].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful runs to rank")
            return None
            
        # Calculate method averages
        method_stats = successful_df.groupby('method').agg({
            'accuracy': 'mean',
            'f1_score': 'mean', 
            'total_training_time': 'mean',
            'time_per_epoch': 'mean',
            'peak_gpu_memory_gb': 'mean',
            'avg_gpu_utilization': 'mean'
        }).round(4)
        
        # Calculate rankings (1 = best, higher = worse)
        rankings = pd.DataFrame(index=method_stats.index)
        
        # For accuracy and F1, higher is better (ascending=False)
        rankings['accuracy_rank'] = method_stats['accuracy'].rank(ascending=False)
        rankings['f1_rank'] = method_stats['f1_score'].rank(ascending=False)
        
        # For time and memory, lower is better (ascending=True)
        rankings['time_rank'] = method_stats['total_training_time'].rank(ascending=True)
        rankings['memory_rank'] = method_stats['peak_gpu_memory_gb'].rank(ascending=True)
        
        # GPU utilization: higher is generally better for efficiency
        rankings['gpu_util_rank'] = method_stats['avg_gpu_utilization'].rank(ascending=False)
        
        # Calculate overall score (lower is better)
        rankings['overall_score'] = rankings.mean(axis=1)
        rankings['overall_rank'] = rankings['overall_score'].rank(ascending=True)
        
        # Create final ranking table
        final_ranking = pd.DataFrame({
            'method': method_stats.index,
            'accuracy': method_stats['accuracy'],
            'f1_score': method_stats['f1_score'],
            'avg_train_time_s': method_stats['total_training_time'],
            'peak_memory_gb': method_stats['peak_gpu_memory_gb'],
            'overall_rank': rankings['overall_rank']
        })
        
        final_ranking = final_ranking.sort_values('overall_rank')
        
        print("🏆 PERFORMANCE RANKING:")
        print("=" * 60)
        print(final_ranking.to_string(index=False, float_format='%.4f'))
        
        return final_ranking

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--results-dir', type=str, default='/home/amin/TimesURL/benchmark_results',
                       help='Directory containing benchmark results')
    parser.add_argument('--plots', action='store_true', help='Generate comparison plots')
    parser.add_argument('--report', action='store_true', help='Generate summary report')
    parser.add_argument('--ranking', action='store_true', help='Show performance ranking')
    parser.add_argument('--all', action='store_true', help='Generate all analyses')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_dir)
    
    # Load data
    df = analyzer.load_all_results()
    if df is None:
        return
        
    # Generate requested analyses
    if args.all or args.plots:
        analyzer.create_comparison_plots()
        
    if args.all or args.report:
        analyzer.generate_summary_report()
        
    if args.all or args.ranking:
        analyzer.create_performance_ranking()
        
    if not any([args.plots, args.report, args.ranking, args.all]):
        print("📊 Quick overview:")
        print(f"   • {len(df)} total benchmark runs")
        print(f"   • {len(df[df['success'] == True])} successful runs")
        print(f"   • {len(df['dataset'].unique())} datasets tested")
        print(f"   • {len(df['method'].unique())} methods compared")
        print("\nUse --all to generate complete analysis")

if __name__ == '__main__':
    main()
