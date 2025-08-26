#!/usr/bin/env python3
"""
Expand Enhanced Baselines Integration
====================================

This script expands the enhanced baselines integration to include more
baseline models (TFC, TS-TCC, SimCLR, etc.) by analyzing their interfaces
and adapting them to work with the TimeHUT enhanced benchmarking system.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add TimeHUT path
sys.path.append('/home/amin/TSlib/models/timehut')

def analyze_baseline_model_interfaces():
    """Analyze baseline model interfaces to understand their command formats"""
    
    baselines_dir = Path('/home/amin/TSlib/models/timehut/baselines')
    analysis = {}
    
    baseline_models = {
        'TFC': {
            'script': 'main.py',
            'expected_args': ['--target_dataset', '--pretrain_dataset', '--training_mode', '--seed'],
            'notes': 'Uses custom dataset naming'
        },
        'TS-TCC': {
            'script': 'main.py', 
            'expected_args': ['--selected_dataset', '--training_mode', '--seed'],
            'notes': 'Uses custom dataset naming'
        },
        'SimCLR': {
            'script': 'train_model.py',
            'expected_args': ['--dataset', '--epochs', '--seed'],
            'notes': 'Standard interface expected'
        },
        'Mixing-up': {
            'script': 'train_model.py',
            'expected_args': ['hardcoded'],
            'notes': 'Needs major adaptation'
        },
        'CLOCS': {
            'script': 'unknown',
            'expected_args': ['unknown'],
            'notes': 'Needs investigation'
        }
    }
    
    print("🔍 Analyzing Baseline Model Interfaces")
    print("="*50)
    
    for model_name, expected in baseline_models.items():
        model_dir = baselines_dir / model_name
        analysis[model_name] = {
            'exists': model_dir.exists(),
            'main_script': None,
            'help_output': '',
            'analysis_status': 'pending'
        }
        
        if model_dir.exists():
            print(f"\n📁 Analyzing {model_name}...")
            
            # Check for main script
            main_script = model_dir / expected['script']
            if main_script.exists():
                analysis[model_name]['main_script'] = str(main_script)
                print(f"   ✅ Main script found: {expected['script']}")
                
                # Try to get help output
                try:
                    result = subprocess.run(
                        ['python', expected['script'], '--help'],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        cwd=str(model_dir)
                    )
                    
                    if result.returncode == 0:
                        analysis[model_name]['help_output'] = result.stdout
                        analysis[model_name]['analysis_status'] = 'help_available'
                        print(f"   ✅ Help output captured")
                    else:
                        analysis[model_name]['analysis_status'] = 'no_help'
                        print(f"   ⚠️  No help available")
                        
                except subprocess.TimeoutExpired:
                    analysis[model_name]['analysis_status'] = 'help_timeout'
                    print(f"   ⚠️  Help command timed out")
                except Exception as e:
                    analysis[model_name]['analysis_status'] = f'error: {e}'
                    print(f"   ❌ Error getting help: {e}")
            else:
                print(f"   ❌ Main script not found: {expected['script']}")
        else:
            print(f"   ❌ Directory not found: {model_dir}")
    
    return analysis

def create_expanded_baselines_config(analysis):
    """Create expanded configuration for baseline models"""
    
    print(f"\n📝 Creating Expanded Baselines Configuration")
    
    # Enhanced configuration with analyzed models
    expanded_config = {
        'TS2vec': {
            'script': 'train.py',
            'args_template': [
                '{dataset}', '{run_name}',
                '--loader', 'UEA',
                '--epochs', '{epochs}',
                '--batch_size', '{batch_size}',
                '--lr', '{lr}',
                '--seed', '{seed}',
                '--train', '--eval'
            ],
            'timeout': 1800,
            'status': 'working',
            'description': 'Enhanced TS2vec with TimeHUT optimizations',
            'supports_uea': True
        }
    }
    
    # Add TFC if available
    if analysis.get('TFC', {}).get('exists'):
        expanded_config['TFC'] = {
            'script': 'main.py',
            'args_template': [
                '--target_dataset', '{dataset_mapped}',
                '--pretrain_dataset', '{dataset_mapped}',
                '--training_mode', 'fine_tune_test',
                '--seed', '{seed}',
                '--epochs', '{epochs}'
            ],
            'timeout': 2400,
            'status': 'needs_dataset_mapping',
            'description': 'TFC with dataset mapping for UEA datasets',
            'supports_uea': False,
            'dataset_mapping': {
                'AtrialFibrillation': 'Epilepsy',  # Similar medical dataset
                'MotorImagery': 'FaceDetection'   # Similar movement dataset
            }
        }
    
    # Add TS-TCC if available
    if analysis.get('TS-TCC', {}).get('exists'):
        expanded_config['TS-TCC'] = {
            'script': 'main.py',
            'args_template': [
                '--selected_dataset', '{dataset_mapped}',
                '--training_mode', 'supervised',
                '--seed', '{seed}',
                '--epochs', '{epochs}'
            ],
            'timeout': 2400,
            'status': 'needs_dataset_mapping',
            'description': 'TS-TCC with dataset mapping for UEA datasets',
            'supports_uea': False,
            'dataset_mapping': {
                'AtrialFibrillation': 'Epilepsy',
                'MotorImagery': 'FaceDetection'
            }
        }
    
    # Add SimCLR if available
    if analysis.get('SimCLR', {}).get('exists'):
        expanded_config['SimCLR'] = {
            'script': 'train_model.py',
            'args_template': [
                '--dataset', '{dataset}',
                '--epochs', '{epochs}',
                '--batch_size', '{batch_size}',
                '--lr', '{lr}',
                '--seed', '{seed}'
            ],
            'timeout': 2400,
            'status': 'needs_testing',
            'description': 'SimCLR adapted for UEA datasets',
            'supports_uea': False
        }
    
    return expanded_config

def create_enhanced_baselines_integration_v2():
    """Create version 2 of enhanced baselines integration with more models"""
    
    print("🚀 Creating Enhanced Baselines Integration V2")
    
    # Analyze existing models
    analysis = analyze_baseline_model_interfaces()
    
    # Create expanded configuration
    expanded_config = create_expanded_baselines_config(analysis)
    
    # Generate enhanced integration script with more models
    script_content = f'''#!/usr/bin/env python3
"""
Enhanced TimeHUT Baselines Integration V2
=========================================

Extended version with support for multiple baseline models:
- TS2vec (Working)
- TFC (With dataset mapping)
- TS-TCC (With dataset mapping) 
- SimCLR (Needs testing)

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# Import the base enhanced integration
sys.path.append('/home/amin/TSlib/models/timehut')
from enhanced_baselines_integration import (
    EnhancedTimeHUTBaselinesIntegrator,
    DATASET_CONFIGS,
    ENHANCED_OPTIMIZATION_FLAGS
)

# Expanded baseline models configuration
EXPANDED_BASELINES_CONFIG = {json.dumps(expanded_config, indent=4)}

class EnhancedTimeHUTBaselinesIntegratorV2(EnhancedTimeHUTBaselinesIntegrator):
    """Extended version with more baseline models"""
    
    def __init__(self):
        super().__init__()
        print("🚀 Enhanced TimeHUT Baselines Integration V2")
        print(f"📊 Expanded to {len(EXPANDED_BASELINES_CONFIG)} baseline models")
    
    def get_expanded_working_baselines(self, dataset_name, config):
        """Get expanded list of working baseline models"""
        
        working_baselines = {{}}
        
        for model_name, model_config in EXPANDED_BASELINES_CONFIG.items():
            if model_config['status'] in ['working', 'needs_testing']:
                
                # Build arguments from template
                args = []
                for arg_template in model_config['args_template']:
                    arg = arg_template.format(
                        dataset=dataset_name,
                        dataset_mapped=model_config.get('dataset_mapping', {{}}).get(dataset_name, dataset_name),
                        run_name=f'enhanced_{{dataset_name}}_{{int(time.time())}}',
                        epochs=config['optimal_epochs'],
                        batch_size=config['optimal_batch_size'],
                        lr=config['learning_rate'],
                        seed=42
                    )
                    args.append(arg)
                
                working_baselines[model_name] = {{
                    'script': model_config['script'],
                    'args': args,
                    'timeout': model_config['timeout'],
                    'description': model_config['description']
                }}
        
        return working_baselines
    
    def run_enhanced_baseline_benchmark_v2(self, dataset_name):
        """Run enhanced benchmark with expanded baseline models"""
        
        print(f"\\n🚀 Enhanced Baseline Benchmark V2: {{dataset_name}}")
        print("="*70)
        
        # Get dataset configuration
        config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['AtrialFibrillation'])
        
        print(f"📊 Dataset Configuration for {{dataset_name}}:")
        for key, value in config.items():
            print(f"   - {{key.replace('_', ' ').title()}}: {{value}}")
        
        # Get expanded working baselines
        working_baselines = self.get_expanded_working_baselines(dataset_name, config)
        
        print(f"\\n🔧 Running {{len(working_baselines)}} baseline models:")
        for model_name in working_baselines.keys():
            status = EXPANDED_BASELINES_CONFIG[model_name]['status']
            print(f"   - {{model_name}} ({{status}})")
        
        results = {{}}
        
        for model_name, model_config in working_baselines.items():
            print(f"\\n🔥 Running Enhanced {{model_name}}...")
            result = self.run_single_baseline_model(
                model_name, dataset_name, model_config, config
            )
            results[model_name] = result
        
        # Generate comprehensive report
        self.generate_enhanced_benchmark_report(dataset_name, results)
        
        return results

    def run_comprehensive_benchmark_v2(self):
        """Run comprehensive benchmark V2 on both datasets"""
        print("\\n🔥 COMPREHENSIVE ENHANCED BASELINE BENCHMARK V2")
        print("="*80)
        
        # Setup prerequisites
        self.setup_dataset_links()
        
        datasets = ['AtrialFibrillation', 'MotorImagery']
        all_results = {{}}
        
        for dataset in datasets:
            print(f"\\n🎯 Enhanced V2 benchmarking all models on {{dataset}}...")
            results = self.run_enhanced_baseline_benchmark_v2(dataset)
            all_results[dataset] = results
        
        # Generate combined analysis
        self.generate_combined_analysis(all_results)
        
        print(f"\\n🎉 Comprehensive enhanced V2 benchmark complete!")
        print(f"📊 Results directory: {{self.results_dir}}")
        
        return all_results

def main():
    """Main function for V2 integration"""
    print("🚀 Starting Enhanced TimeHUT Baselines Integration V2...")
    
    integrator = EnhancedTimeHUTBaselinesIntegratorV2()
    
    # Run comprehensive benchmark V2
    results = integrator.run_comprehensive_benchmark_v2()
    
    print("\\n✅ Enhanced baselines integration V2 complete!")
    return results

if __name__ == "__main__":
    main()
'''
    
    # Save the V2 script
    v2_script_path = Path('/home/amin/TSlib/models/timehut/enhanced_baselines_integration_v2.py')
    with open(v2_script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Enhanced Baselines Integration V2 created: {v2_script_path}")
    
    # Save analysis results
    analysis_path = Path('/home/amin/TSlib/models/timehut/baseline_models_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"📄 Baseline analysis saved: {analysis_path}")
    
    return v2_script_path

def main():
    """Main function"""
    print("🔧 Expanding Enhanced TimeHUT Baselines Integration")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        v2_script = create_enhanced_baselines_integration_v2()
        
        print("\\n✅ Expansion completed successfully!")
        print("\\n📝 What's New in V2:")
        print("   - Extended support for TFC, TS-TCC, SimCLR")
        print("   - Dataset mapping for models that don't support UEA directly")
        print("   - Enhanced configuration system")
        print("   - Backward compatibility with V1")
        
        print("\\n📝 Next Steps:")
        print("   1. Test V2: python test_enhanced_baselines.py")
        print("   2. Run V2 benchmark: python enhanced_baselines_integration_v2.py")
        print("   3. Compare V1 vs V2 results")
        
    except Exception as e:
        print(f"❌ Expansion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
