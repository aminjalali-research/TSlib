"""
Temperature Scheduling Methods for TimeHUT
Implementation of various temperature scheduling strategies for contrastive learning
"""

import torch
import numpy as np
import math
import random
import time

class TemperatureScheduler:
    """
    Various temperature scheduling methods for contrastive learning
    """
    
    # Class variable for momentum-based scheduler
    _momentum_term = 0.0
    
    @staticmethod
    def cosine_annealing(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, phase=0.0, frequency=1.0, bias=0.0):
        """
        Enhanced cosine annealing schedule with phase control and frequency modulation
        tau = min_tau + 0.5 * (max_tau - min_tau) * (1 + cos(2π * frequency * epoch / t_max + phase)) + bias
        
        Args:
            epoch: Current epoch
            min_tau: Minimum temperature value
            max_tau: Maximum temperature value  
            t_max: Period for one complete cycle
            phase: Phase shift (0 to 2π) - shifts the cosine curve
            frequency: Frequency multiplier - controls oscillation speed
            bias: Additional bias term for asymmetric scheduling
        """
        cosine_term = torch.cos(torch.tensor(2 * np.pi * frequency * epoch / t_max + phase))
        tau = min_tau + 0.5 * (max_tau - min_tau) * (1 + cosine_term) + bias
        return torch.clamp(tau, min_tau, max_tau)  # Ensure bounds
    
    @staticmethod
    def linear_decay(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5):
        """
        Linear decay from max_tau to min_tau
        tau = max_tau - (max_tau - min_tau) * (epoch / t_max)
        """
        if epoch >= t_max:
            return min_tau
        return max_tau - (max_tau - min_tau) * (epoch / t_max)
    
    @staticmethod
    def exponential_decay(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, decay_rate=0.95):
        """
        Exponential decay schedule
        tau = max(min_tau, max_tau * decay_rate^epoch)
        """
        tau = max_tau * (decay_rate ** epoch)
        return max(min_tau, tau)
    
    @staticmethod
    def step_decay(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, step_size=None, gamma=0.5):
        """
        Step decay schedule (reduce by gamma every step_size epochs)
        """
        if step_size is None:
            step_size = int(t_max / 3)  # 3 steps by default
        
        tau = max_tau * (gamma ** (epoch // step_size))
        return max(min_tau, tau)
    
    @staticmethod
    def polynomial_decay(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, power=2.0):
        """
        Polynomial decay schedule
        tau = min_tau + (max_tau - min_tau) * (1 - epoch/t_max)^power
        """
        if epoch >= t_max:
            return min_tau
        return min_tau + (max_tau - min_tau) * ((1 - epoch / t_max) ** power)
    
    @staticmethod
    def sigmoid_decay(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, steepness=1.0):
        """
        Sigmoid-based decay schedule
        tau = min_tau + (max_tau - min_tau) / (1 + exp(steepness * (epoch - t_max/2)))
        """
        sigmoid_val = 1 / (1 + math.exp(steepness * (epoch - t_max / 2)))
        return min_tau + (max_tau - min_tau) * sigmoid_val
    
    @staticmethod
    def warmup_cosine(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, warmup_epochs=2):
        """
        Warmup followed by cosine annealing
        First warmup_epochs: linear increase from min_tau to max_tau
        Then: cosine annealing from max_tau to min_tau
        """
        if epoch < warmup_epochs:
            # Linear warmup
            return min_tau + (max_tau - min_tau) * (epoch / warmup_epochs)
        else:
            # Cosine annealing
            adjusted_epoch = epoch - warmup_epochs
            adjusted_t_max = t_max - warmup_epochs
            return min_tau + 0.5 * (max_tau - min_tau) * (1 + math.cos(np.pi * adjusted_epoch / adjusted_t_max))
    
    @staticmethod
    def constant_temperature(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5):
        """
        Constant temperature (baseline)
        """
        return max_tau
    
    @staticmethod
    def cyclic_temperature(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, cycle_length=None):
        """
        Cyclic temperature schedule (sawtooth pattern)
        """
        if cycle_length is None:
            cycle_length = t_max / 3  # 3 cycles by default
        
        cycle_pos = (epoch % cycle_length) / cycle_length
        return min_tau + (max_tau - min_tau) * cycle_pos

    @staticmethod
    def no_scheduling(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5):
        """
        No temperature scheduling (constant tau = 1.0)
        This replicates the original TimeHUT behavior
        """
        return 1.0

    @staticmethod
    def adaptive_cosine_annealing(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, 
                                 momentum=0.9, adaptation_rate=0.1, performance_history=None):
        """
        Adaptive cosine annealing that adjusts based on training performance
        Combines cosine scheduling with momentum and performance-based adaptation
        
        Args:
            epoch: Current epoch
            min_tau: Minimum temperature value
            max_tau: Maximum temperature value
            t_max: Period for one complete cycle
            momentum: Momentum factor for smooth transitions
            adaptation_rate: Rate of adaptation based on performance
            performance_history: List of recent performance metrics (e.g., loss values)
        """
        # Base cosine schedule
        base_tau = min_tau + 0.5 * (max_tau - min_tau) * (1 + torch.cos(torch.tensor(np.pi * epoch / t_max)))
        
        # Adaptive component based on performance
        if performance_history and len(performance_history) > 1:
            # If performance is improving, allow more aggressive scheduling
            recent_trend = performance_history[-1] - performance_history[-2]
            adaptation = -adaptation_rate * recent_trend  # Negative because lower loss is better
            adapted_range = (max_tau - min_tau) * (1 + adaptation)
            base_tau = min_tau + 0.5 * adapted_range * (1 + torch.cos(torch.tensor(np.pi * epoch / t_max)))
        
        return torch.clamp(base_tau, min_tau * 0.5, max_tau * 1.2)  # Allow some flexibility

    @staticmethod
    def multi_cycle_cosine(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, 
                          num_cycles=3, decay_factor=0.8):
        """
        Multi-cycle cosine annealing with decreasing amplitude
        Each cycle has smaller amplitude than the previous one
        
        Args:
            epoch: Current epoch
            min_tau: Minimum temperature value
            max_tau: Maximum temperature value
            t_max: Total training period
            num_cycles: Number of cosine cycles
            decay_factor: Factor by which amplitude decays each cycle
        """
        cycle_length = t_max / num_cycles
        current_cycle = int(epoch // cycle_length)
        cycle_epoch = epoch % cycle_length
        
        # Decay amplitude with each cycle
        current_amplitude = (max_tau - min_tau) * (decay_factor ** current_cycle)
        current_max = min_tau + current_amplitude
        
        cosine_val = torch.cos(torch.tensor(np.pi * cycle_epoch / cycle_length))
        tau = min_tau + 0.5 * current_amplitude * (1 + cosine_val)
        
        return torch.clamp(tau, min_tau, max_tau)

    @staticmethod
    def cosine_with_restarts(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, 
                           restart_period=5.0, restart_mult=1.5):
        """
        Cosine annealing with warm restarts (SGDR-style for temperature)
        Periodically restarts the cosine annealing to escape local minima
        
        Args:
            epoch: Current epoch
            min_tau: Minimum temperature value
            max_tau: Maximum temperature value
            t_max: Initial restart period
            restart_period: Base period between restarts
            restart_mult: Multiplier for restart period after each restart
        """
        # Calculate which restart cycle we're in
        current_period = restart_period
        total_epochs = 0
        cycle = 0
        
        while total_epochs + current_period <= epoch:
            total_epochs += current_period
            current_period *= restart_mult
            cycle += 1
        
        # Position within current cycle
        cycle_epoch = epoch - total_epochs
        progress = cycle_epoch / current_period
        
        # Cosine annealing within current cycle
        cosine_val = torch.cos(torch.tensor(np.pi * progress))
        tau = min_tau + 0.5 * (max_tau - min_tau) * (1 + cosine_val)
        
        return tau

    # =====================================================
    # NOVEL EFFICIENT SCHEDULERS
    # =====================================================
    
    @staticmethod
    def momentum_adaptive_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5,
                                   performance_history=None, momentum=0.9, adapt_rate=0.1):
        """
        Novel momentum-based adaptive scheduler that adjusts based on training progress
        Uses momentum from accuracy improvements to modulate temperature changes
        """
        base_tau = min_tau + (max_tau - min_tau) * (1 - epoch / t_max) if epoch < t_max else min_tau
        
        if performance_history is None or len(performance_history) < 2:
            return base_tau
        
        # Calculate momentum-weighted improvement
        recent_improvement = performance_history[-1] - performance_history[-2] if len(performance_history) >= 2 else 0
        
        # Use class variable for momentum term
        if not hasattr(TemperatureScheduler, '_momentum_term'):
            TemperatureScheduler._momentum_term = recent_improvement
        else:
            TemperatureScheduler._momentum_term = momentum * TemperatureScheduler._momentum_term + (1-momentum) * recent_improvement
        
        # Adjust temperature based on momentum
        tau_adjustment = adapt_rate * TemperatureScheduler._momentum_term
        adjusted_tau = base_tau + tau_adjustment
        
        return torch.clamp(torch.tensor(adjusted_tau), min_tau, max_tau).item()

    @staticmethod  
    def triangular_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, cycle_length=None):
        """
        Triangular wave scheduler inspired by Cyclic Learning Rates (CLR)
        Creates triangular waves between min_tau and max_tau
        """
        if cycle_length is None:
            cycle_length = t_max / 3
        
        cycle_position = epoch % cycle_length
        if cycle_position <= cycle_length / 2:
            # Ascending phase
            tau = min_tau + (max_tau - min_tau) * (2 * cycle_position / cycle_length)
        else:
            # Descending phase  
            tau = max_tau - (max_tau - min_tau) * (2 * (cycle_position - cycle_length/2) / cycle_length)
        
        return tau

    @staticmethod
    def onecycle_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, 
                          peak_epoch_ratio=0.3, final_tau_ratio=0.1):
        """
        OneCycle scheduler for temperature (inspired by OneCycle LR)
        Single cycle: start low -> peak high -> end very low
        Designed for superconvergence in contrastive learning
        """
        peak_epoch = t_max * peak_epoch_ratio
        final_tau = min_tau * final_tau_ratio
        
        if epoch <= peak_epoch:
            # Phase 1: Increase from min_tau to max_tau
            progress = epoch / peak_epoch
            tau = min_tau + (max_tau - min_tau) * progress
        else:
            # Phase 2: Decrease from max_tau to final_tau
            progress = (epoch - peak_epoch) / (t_max - peak_epoch)
            tau = max_tau - (max_tau - final_tau) * progress
        
        return max(final_tau, tau)

    @staticmethod
    def hyperbolic_tangent_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, 
                                   steepness=2.0, shift=0.5):
        """
        Hyperbolic tangent scheduler for smooth S-curve transition
        Provides natural saturation and smooth gradients
        """
        # Normalize epoch to [-1, 1] range around shift point
        normalized_epoch = steepness * (epoch / t_max - shift)
        
        # Apply tanh for smooth S-curve
        tanh_val = math.tanh(normalized_epoch)
        
        # Map tanh output [-1, 1] to [max_tau, min_tau] (inverted for cooling)
        tau = min_tau + 0.5 * (max_tau - min_tau) * (1 - tanh_val)
        
        return tau

    @staticmethod  
    def logarithmic_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5, 
                             log_base=math.e, offset=1.0):
        """
        Logarithmic decay scheduler for gentle, long-tail reduction
        Natural for contrastive learning convergence
        """
        if epoch >= t_max:
            return min_tau
        
        # Logarithmic decay
        log_factor = math.log(epoch + offset, log_base) / math.log(t_max + offset, log_base)
        tau = max_tau - (max_tau - min_tau) * log_factor
        
        return max(min_tau, tau)

    @staticmethod
    def piecewise_linear_plateau_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5,
                                         breakpoints=None, plateau_lengths=None):
        """
        Piecewise linear scheduler with plateau periods for stability
        Multiple linear segments with flat plateaus
        """
        if breakpoints is None:
            breakpoints = [t_max * 0.2, t_max * 0.5, t_max * 0.8]
        if plateau_lengths is None:
            plateau_lengths = [t_max * 0.1, t_max * 0.1, t_max * 0.1]
        
        # Define temperature values at breakpoints
        temp_values = [max_tau, max_tau * 0.7, max_tau * 0.4, min_tau]
        
        current_segment = 0
        cumulative_time = 0
        
        for i, (bp, plateau_len) in enumerate(zip(breakpoints, plateau_lengths)):
            if epoch <= cumulative_time + bp:
                # Linear interpolation within segment
                segment_progress = (epoch - cumulative_time) / bp
                tau = temp_values[i] + (temp_values[i+1] - temp_values[i]) * segment_progress
                return tau
            
            cumulative_time += bp
            
            # Check if in plateau
            if epoch <= cumulative_time + plateau_len:
                return temp_values[i+1]
            
            cumulative_time += plateau_len
        
        return min_tau

    @staticmethod
    def inverse_time_decay_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5,
                                   decay_rate=0.1, staircase=False):
        """
        Inverse time decay: tau = initial_tau / (1 + decay_rate * epoch)
        Mathematically proven convergence properties from optimization theory
        """
        if staircase:
            # Step-wise inverse time decay
            step_size = max(1, int(t_max / 10))
            effective_epoch = (epoch // step_size) * step_size
        else:
            effective_epoch = epoch
        
        tau = max_tau / (1 + decay_rate * effective_epoch)
        return max(min_tau, tau)

    @staticmethod
    def double_exponential_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5,
                                   fast_decay=0.1, slow_decay=0.01, transition_epoch=None):
        """
        Double exponential (LaPlace-style) scheduler
        Rapid initial decay then slow fine-tuning phase
        """
        if transition_epoch is None:
            transition_epoch = t_max * 0.3
        
        if epoch <= transition_epoch:
            # Fast decay phase
            decay_factor = math.exp(-fast_decay * epoch)
            tau = min_tau + (max_tau - min_tau) * decay_factor
        else:
            # Slow decay phase
            transition_tau = min_tau + (max_tau - min_tau) * math.exp(-fast_decay * transition_epoch)
            remaining_epochs = epoch - transition_epoch
            slow_decay_factor = math.exp(-slow_decay * remaining_epochs)
            tau = min_tau + (transition_tau - min_tau) * slow_decay_factor
        
        return max(min_tau, tau)

    @staticmethod
    def noisy_cosine_scheduler(epoch, min_tau=0.15, max_tau=0.75, t_max=10.5,
                              noise_factor=0.05, noise_decay=0.95):
        """
        Cosine annealing with decaying noise for exploration
        Adds controlled randomness that decreases over time
        """
        # Base cosine schedule
        base_tau = min_tau + 0.5 * (max_tau - min_tau) * (1 + math.cos(math.pi * epoch / t_max))
        
        # Add decaying noise
        noise_magnitude = noise_factor * (noise_decay ** epoch) * (max_tau - min_tau)
        noise = random.uniform(-noise_magnitude, noise_magnitude)
        
        tau = base_tau + noise
        return torch.clamp(torch.tensor(tau), min_tau, max_tau).item()

    @staticmethod
    def get_scheduler(method_name):
        """
        Factory method to get scheduler function by name
        """
        schedulers = {
            # Original schedulers
            'cosine_annealing': TemperatureScheduler.cosine_annealing,
            'linear_decay': TemperatureScheduler.linear_decay,
            'exponential_decay': TemperatureScheduler.exponential_decay,
            'step_decay': TemperatureScheduler.step_decay,
            'polynomial_decay': TemperatureScheduler.polynomial_decay,
            'sigmoid_decay': TemperatureScheduler.sigmoid_decay,
            'warmup_cosine': TemperatureScheduler.warmup_cosine,
            'constant': TemperatureScheduler.constant_temperature,
            'cyclic': TemperatureScheduler.cyclic_temperature,
            'no_scheduling': TemperatureScheduler.no_scheduling,
            'adaptive_cosine_annealing': TemperatureScheduler.adaptive_cosine_annealing,
            'multi_cycle_cosine': TemperatureScheduler.multi_cycle_cosine,
            'cosine_with_restarts': TemperatureScheduler.cosine_with_restarts,
            
            # Novel efficient schedulers
            'momentum_adaptive': TemperatureScheduler.momentum_adaptive_scheduler,
            'triangular': TemperatureScheduler.triangular_scheduler,
            'onecycle': TemperatureScheduler.onecycle_scheduler,
            'hyperbolic_tangent': TemperatureScheduler.hyperbolic_tangent_scheduler,
            'logarithmic': TemperatureScheduler.logarithmic_scheduler,
            'piecewise_linear_plateau': TemperatureScheduler.piecewise_linear_plateau_scheduler,
            'inverse_time_decay': TemperatureScheduler.inverse_time_decay_scheduler,
            'double_exponential': TemperatureScheduler.double_exponential_scheduler,
            'noisy_cosine': TemperatureScheduler.noisy_cosine_scheduler,
        }
        
        if method_name not in schedulers:
            raise ValueError(f"Unknown scheduler: {method_name}. Available: {list(schedulers.keys())}")
        
        return schedulers[method_name]

def create_temperature_settings(method='cosine_annealing', min_tau=0.15, max_tau=0.75, t_max=10.5, **kwargs):
    """
    Create temperature settings dictionary for different scheduling methods
    """
    settings = {
        'method': method,
        'min_tau': min_tau,
        'max_tau': max_tau,
        't_max': t_max,
        'scheduler_func': TemperatureScheduler.get_scheduler(method)
    }
    
    # Add method-specific parameters
    if method == 'cosine_annealing':
        settings['phase'] = kwargs.get('phase', 0.0)
        settings['frequency'] = kwargs.get('frequency', 1.0) 
        settings['bias'] = kwargs.get('bias', 0.0)
    elif method == 'adaptive_cosine_annealing':
        settings['momentum'] = kwargs.get('momentum', 0.9)
        settings['adaptation_rate'] = kwargs.get('adaptation_rate', 0.1)
        settings['performance_history'] = kwargs.get('performance_history', None)
    elif method == 'multi_cycle_cosine':
        settings['num_cycles'] = kwargs.get('num_cycles', 3)
        settings['decay_factor'] = kwargs.get('decay_factor', 0.8)
    elif method == 'cosine_with_restarts':
        settings['restart_period'] = kwargs.get('restart_period', 5.0)
        settings['restart_mult'] = kwargs.get('restart_mult', 1.5)
    elif method == 'exponential_decay':
        settings['decay_rate'] = kwargs.get('decay_rate', 0.95)
    elif method == 'step_decay':
        settings['step_size'] = kwargs.get('step_size', int(t_max / 3))
        settings['gamma'] = kwargs.get('gamma', 0.5)
    elif method == 'polynomial_decay':
        settings['power'] = kwargs.get('power', 2.0)
    elif method == 'sigmoid_decay':
        settings['steepness'] = kwargs.get('steepness', 1.0)
    elif method == 'warmup_cosine':
        settings['warmup_epochs'] = kwargs.get('warmup_epochs', 2)
    elif method == 'cyclic':
        settings['cycle_length'] = kwargs.get('cycle_length', t_max / 3)
        settings['adaptation_rate'] = kwargs.get('adaptation_rate', 0.1)
    elif method == 'multi_cycle_cosine':
        settings['num_cycles'] = kwargs.get('num_cycles', 3)
        settings['decay_factor'] = kwargs.get('decay_factor', 0.8)
    elif method == 'cosine_with_restarts':
        settings['restart_period'] = kwargs.get('restart_period', 5.0)
        settings['restart_mult'] = kwargs.get('restart_mult', 1.5)
    
    return settings

# =====================================================
# PYHOPPER OPTIMIZATION INTEGRATION
# =====================================================

try:
    import pyhopper
    import subprocess
    import json
    PYHOPPER_AVAILABLE = True
except ImportError:
    PYHOPPER_AVAILABLE = False

class SchedulerOptimizer:
    """
    PyHopper optimization for temperature schedulers
    Automatically optimize hyperparameters for enhanced temperature scheduling variants
    """
    
    def __init__(self, base_dir='/home/amin/TSlib/models/timehut', dataset='Chinatown'):
        self.base_dir = base_dir
        self.dataset = dataset
        self.results = []
    
    def run_training_command(self, params, method_name, dataset=None):
        """Run training with given parameters and return accuracy"""
        if dataset is None:
            dataset = self.dataset
            
        # Base command
        cmd = [
            'python', 'train_unified_comprehensive.py',
            dataset, f'{method_name}_optimized',
            '--loader', 'UCR',
            '--gpu', '0',
            '--batch-size', '8',
            '--epochs', '200',
            '--seed', '2002',
            '--method', 'acc',
            '--dataroot', '/home/amin/TSlib/datasets',
            '--scenario', 'amc_temp'
        ]
        
        # Add scheduler-specific parameters
        if method_name == 'cosine_annealing':
            cmd.extend([
                '--temp-method', 'cosine_annealing',
                '--min-tau', str(params['min_tau']),
                '--max-tau', str(params['max_tau']),
                '--t-max', str(params['t_max'])
            ])
            # Enhanced cosine annealing parameters would need to be added to the training script
            # For now, use the basic version
            
        elif method_name == 'multi_cycle_cosine':
            cmd.extend([
                '--temp-method', 'multi_cycle_cosine',
                '--min-tau', str(params['min_tau']),
                '--max-tau', str(params['max_tau']),
                '--t-max', str(params['t_max']),
                '--temp-num-cycles', str(int(params['num_cycles'])),
                '--temp-decay-factor', str(params['decay_factor'])
            ])
            
        elif method_name == 'adaptive_cosine_annealing':
            cmd.extend([
                '--temp-method', 'adaptive_cosine_annealing',
                '--min-tau', str(params['min_tau']),
                '--max-tau', str(params['max_tau']),
                '--t-max', str(params['t_max'])
            ])
            # Momentum and adaptation rate would need to be added to training script
            
        elif method_name == 'cosine_with_restarts':
            cmd.extend([
                '--temp-method', 'cosine_with_restarts',
                '--min-tau', str(params['min_tau']),
                '--max-tau', str(params['max_tau']),
                '--t-max', str(params['t_max'])
            ])
            # Restart parameters would need to be added to training script
            
        elif method_name == 'step_decay':
            cmd.extend([
                '--temp-method', 'step_decay',
                '--min-tau', str(params['min_tau']),
                '--max-tau', str(params['max_tau']),
                '--t-max', str(params['t_max']),
                '--temp-step-size', str(int(params.get('step_size', params['t_max'] / 3))),
                '--temp-gamma', str(params.get('gamma', 0.5))
            ])
            
        elif method_name == 'exponential_decay':
            cmd.extend([
                '--temp-method', 'exponential_decay',
                '--min-tau', str(params['min_tau']),
                '--max-tau', str(params['max_tau']),
                '--t-max', str(params['t_max']),
                '--temp-decay-rate', str(params.get('decay_rate', 0.95))
            ])
        
        try:
            # Run the training command
            result = subprocess.run(
                cmd, 
                cwd=self.base_dir,
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"Training failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return 0.0
            
            # Parse accuracy from output
            lines = result.stdout.split('\n')
            for line in reversed(lines):
                if 'Accuracy:' in line:
                    try:
                        acc_str = line.split('Accuracy:')[1].strip()
                        accuracy = float(acc_str)
                        return accuracy
                    except (ValueError, IndexError):
                        continue
            
            print("Could not find accuracy in output")
            return 0.0
            
        except subprocess.TimeoutExpired:
            print("Training timed out")
            return 0.0
        except Exception as e:
            print(f"Error running training: {e}")
            return 0.0
    
    def optimize_cosine_annealing(self, steps=20):
        """Optimize enhanced cosine annealing parameters"""
        if not PYHOPPER_AVAILABLE:
            return {'error': 'PyHopper not available'}
            
        print("🔍 Optimizing Enhanced Cosine Annealing...")
        
        search_space = pyhopper.Search({
            "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
            "max_tau": pyhopper.float(0.6, 0.9, "0.02f"), 
            "t_max": pyhopper.float(15, 35, "1.0f")
        })
        
        def objective(params):
            return self.run_training_command(params, 'cosine_annealing')
        
        best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
        
        return {
            'method': 'cosine_annealing',
            'best_params': best_params,
            'best_accuracy': objective(best_params)
        }
    
    def optimize_multi_cycle_cosine(self, steps=25):
        """Optimize multi-cycle cosine annealing parameters"""
        if not PYHOPPER_AVAILABLE:
            return {'error': 'PyHopper not available'}
            
        print("🔍 Optimizing Multi-Cycle Cosine Annealing...")
        
        search_space = pyhopper.Search({
            "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
            "max_tau": pyhopper.float(0.6, 0.9, "0.02f"),
            "t_max": pyhopper.float(15, 35, "1.0f"), 
            "num_cycles": pyhopper.int(2, 8),  # 2 to 8 cycles
            "decay_factor": pyhopper.float(0.6, 0.95, "0.05f")
        })
        
        def objective(params):
            return self.run_training_command(params, 'multi_cycle_cosine')
        
        best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
        
        return {
            'method': 'multi_cycle_cosine', 
            'best_params': best_params,
            'best_accuracy': objective(best_params)
        }
    
    def optimize_step_decay(self, steps=20):
        """Optimize step decay scheduler parameters"""
        if not PYHOPPER_AVAILABLE:
            return {'error': 'PyHopper not available'}
            
        print("🔍 Optimizing Step Decay Scheduler...")
        
        search_space = pyhopper.Search({
            "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
            "max_tau": pyhopper.float(0.6, 0.9, "0.02f"),
            "t_max": pyhopper.float(15, 35, "1.0f"),
            "step_size": pyhopper.int(3, 12),
            "gamma": pyhopper.float(0.3, 0.8, "0.05f")
        })
        
        def objective(params):
            return self.run_training_command(params, 'step_decay')
        
        best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
        
        return {
            'method': 'step_decay',
            'best_params': best_params,
            'best_accuracy': objective(best_params)
        }
    
    def optimize_exponential_decay(self, steps=20):
        """Optimize exponential decay scheduler parameters"""
        if not PYHOPPER_AVAILABLE:
            return {'error': 'PyHopper not available'}
            
        print("🔍 Optimizing Exponential Decay Scheduler...")
        
        search_space = pyhopper.Search({
            "min_tau": pyhopper.float(0.01, 0.15, "0.01f"),
            "max_tau": pyhopper.float(0.6, 0.9, "0.02f"),
            "t_max": pyhopper.float(15, 35, "1.0f"),
            "decay_rate": pyhopper.float(0.85, 0.99, "0.01f")
        })
        
        def objective(params):
            return self.run_training_command(params, 'exponential_decay')
        
        best_params = search_space.run(objective, direction="maximize", steps=steps, n_jobs=1)
        
        return {
            'method': 'exponential_decay',
            'best_params': best_params,
            'best_accuracy': objective(best_params)
        }
    
    def run_comprehensive_optimization(self, methods=None):
        """Run comprehensive optimization for multiple enhanced schedulers"""
        if not PYHOPPER_AVAILABLE:
            return {'error': 'PyHopper not available'}
            
        if methods is None:
            methods = [
                ('cosine_annealing', self.optimize_cosine_annealing, 20),
                ('multi_cycle_cosine', self.optimize_multi_cycle_cosine, 25),
                ('step_decay', self.optimize_step_decay, 20),
                ('exponential_decay', self.optimize_exponential_decay, 20)
            ]
        
        print("🚀 Starting Comprehensive Scheduler Optimization...")
        print("=" * 60)
        
        results = []
        start_time = time.time()
        
        for method_name, optimizer_func, steps in methods:
            print(f"\n📊 Optimizing {method_name}...")
            method_start = time.time()
            
            try:
                result = optimizer_func(steps)
                result['optimization_time'] = time.time() - method_start
                results.append(result)
                
                if 'best_accuracy' in result:
                    print(f"✅ {method_name} optimization completed!")
                    print(f"   Best Accuracy: {result['best_accuracy']:.4f}")
                    print(f"   Time: {result['optimization_time']:.1f}s")
                else:
                    print(f"❌ {method_name} optimization failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Error optimizing {method_name}: {e}")
                results.append({
                    'method': method_name,
                    'error': str(e),
                    'optimization_time': time.time() - method_start
                })
        
        # Save results
        total_time = time.time() - start_time
        optimization_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_optimization_time': total_time,
            'dataset': self.dataset,
            'results': results
        }
        
        output_file = f'{self.base_dir}/scheduler_optimization_results.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏆 OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        successful_results = [r for r in results if 'best_accuracy' in r]
        if successful_results:
            best_method = max(successful_results, key=lambda x: x['best_accuracy'])
            
            print(f"🥇 Best Method: {best_method['method']}")
            print(f"🎯 Best Accuracy: {best_method['best_accuracy']:.4f}")
            print(f"⚡ Best Parameters:")
            for param, value in best_method['best_params'].items():
                if isinstance(value, float):
                    print(f"   {param}: {value:.3f}")
                else:
                    print(f"   {param}: {value}")
            
            print(f"\n📊 All Results:")
            for result in successful_results:
                print(f"   {result['method']}: {result['best_accuracy']:.4f}")
        
        print(f"\n⏱️ Total Time: {total_time:.1f}s")
        
        return optimization_results

# =====================================================
# SCHEDULER COMPARISON AND BENCHMARKING
# =====================================================

class SchedulerBenchmark:
    """
    Benchmark different temperature schedulers with systematic comparison
    """
    
    def __init__(self, base_dir='/home/amin/TSlib/models/timehut', dataset='Chinatown'):
        self.base_dir = base_dir
        self.dataset = dataset
        self.benchmark_results = []
    
    def run_scheduler_comparison(self, schedulers=None, epochs=100, trials=3):
        """
        Run systematic comparison of different temperature schedulers
        
        Args:
            schedulers: List of scheduler names to compare
            epochs: Number of training epochs
            trials: Number of trials per scheduler for statistical significance
        """
        if schedulers is None:
            schedulers = [
                'cosine_annealing', 'linear_decay', 'exponential_decay', 'step_decay',
                'polynomial_decay', 'sigmoid_decay', 'warmup_cosine', 'constant',
                'cyclic', 'multi_cycle_cosine', 'cosine_with_restarts'
            ]
        
        print("🔥 Starting Scheduler Benchmark Comparison")
        print("=" * 60)
        
        results = {}
        
        for scheduler in schedulers:
            print(f"\n📊 Benchmarking scheduler: {scheduler}")
            scheduler_results = []
            
            for trial in range(trials):
                print(f"   Trial {trial + 1}/{trials}...")
                
                # Run training command
                cmd = [
                    'python', 'train_unified_comprehensive.py',
                    self.dataset, f'benchmark_{scheduler}_trial_{trial}',
                    '--loader', 'UCR',
                    '--scenario', 'amc_temp',
                    '--temp-method', scheduler,
                    '--epochs', str(epochs),
                    '--batch-size', '8',
                    '--seed', str(2002 + trial),  # Different seed per trial
                    '--verbose'
                ]
                
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=self.base_dir,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    # Parse accuracy
                    accuracy = 0.0
                    for line in reversed(result.stdout.split('\n')):
                        if 'Accuracy:' in line:
                            try:
                                accuracy = float(line.split('Accuracy:')[1].strip())
                                break
                            except (ValueError, IndexError):
                                continue
                    
                    scheduler_results.append({
                        'trial': trial,
                        'accuracy': accuracy,
                        'status': 'success'
                    })
                    
                    print(f"      Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"      Failed: {e}")
                    scheduler_results.append({
                        'trial': trial,
                        'accuracy': 0.0,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Calculate statistics
            successful_results = [r for r in scheduler_results if r['status'] == 'success']
            if successful_results:
                accuracies = [r['accuracy'] for r in successful_results]
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                
                results[scheduler] = {
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'max_accuracy': max(accuracies),
                    'min_accuracy': min(accuracies),
                    'successful_trials': len(successful_results),
                    'total_trials': trials,
                    'raw_results': scheduler_results
                }
                
                print(f"   📈 {scheduler}: {mean_acc:.4f} ± {std_acc:.4f} (max: {max(accuracies):.4f})")
            else:
                results[scheduler] = {
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'max_accuracy': 0.0,
                    'min_accuracy': 0.0,
                    'successful_trials': 0,
                    'total_trials': trials,
                    'raw_results': scheduler_results
                }
                print(f"   ❌ {scheduler}: All trials failed")
        
        # Save benchmark results
        benchmark_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': self.dataset,
            'epochs': epochs,
            'trials_per_scheduler': trials,
            'results': results
        }
        
        output_file = f'{self.base_dir}/scheduler_benchmark_results.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(benchmark_summary, f, indent=2)
            print(f"\n💾 Benchmark results saved to: {output_file}")
        except Exception as e:
            print(f"⚠️ Could not save benchmark results: {e}")
        
        # Print final summary
        self._print_benchmark_summary(results)
        
        return benchmark_summary
    
    def _print_benchmark_summary(self, results):
        """Print formatted benchmark summary"""
        print("\n" + "=" * 60)
        print("🏆 SCHEDULER BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Sort by mean accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
        
        print(f"{'Rank':<4} {'Scheduler':<25} {'Mean Acc':<10} {'Std':<8} {'Max':<8} {'Trials':<7}")
        print("-" * 60)
        
        for rank, (scheduler, result) in enumerate(sorted_results, 1):
            if result['successful_trials'] > 0:
                print(f"{rank:<4} {scheduler:<25} {result['mean_accuracy']:.4f}    "
                      f"{result['std_accuracy']:.4f}   {result['max_accuracy']:.4f}   "
                      f"{result['successful_trials']}/{result['total_trials']}")
            else:
                print(f"{rank:<4} {scheduler:<25} {'FAILED':<10} {'N/A':<8} {'N/A':<8} "
                      f"{result['successful_trials']}/{result['total_trials']}")
        
        if sorted_results and sorted_results[0][1]['successful_trials'] > 0:
            best_scheduler, best_result = sorted_results[0]
            print(f"\n🥇 Winner: {best_scheduler}")
            print(f"🎯 Best Performance: {best_result['mean_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}")

# Test the schedulers and optimization
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    # Test scheduler visualization
    print("🎨 Generating scheduler visualization...")
    epochs = np.arange(0, 20)
    methods = ['cosine_annealing', 'linear_decay', 'exponential_decay', 'step_decay', 
              'polynomial_decay', 'sigmoid_decay', 'warmup_cosine', 'constant', 'cyclic', 'no_scheduling',
              'adaptive_cosine_annealing', 'multi_cycle_cosine', 'cosine_with_restarts']
    
    plt.figure(figsize=(15, 10))
    for i, method in enumerate(methods):
        scheduler = TemperatureScheduler.get_scheduler(method)
        temps = [scheduler(epoch) for epoch in epochs]
        
        plt.subplot(4, 4, i+1)
        plt.plot(epochs, temps, 'b-', linewidth=2)
        plt.title(f'{method.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Temperature (τ)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('temperature_schedulers_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Scheduler visualization saved as 'temperature_schedulers_comparison.png'")
    
    # Test optimization if PyHopper is available
    if PYHOPPER_AVAILABLE:
        print("\n🔧 Testing scheduler optimization...")
        optimizer = SchedulerOptimizer()
        
        # Quick test of one scheduler
        print("Testing single scheduler optimization...")
        # This would run actual training - commented out for safety
        # result = optimizer.optimize_step_decay(steps=3)
        # print(f"Optimization result: {result}")
        
        print("✅ Optimization framework ready")
        print("To run optimization, use:")
        print("  optimizer = SchedulerOptimizer()")
        print("  results = optimizer.run_comprehensive_optimization()")
        
    else:
        print("⚠️ PyHopper not available - optimization features disabled")
    
    # Test benchmarking framework
    print("\n🏁 Benchmarking framework ready")
    print("To run benchmark, use:")
    print("  benchmark = SchedulerBenchmark()")
    print("  results = benchmark.run_scheduler_comparison(trials=3)")
    
    plt.show()
