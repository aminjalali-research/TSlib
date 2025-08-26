"""
Temperature Scheduling Methods for TimeHUT
Implementation of various temperature scheduling strategies for contrastive learning
"""

import torch
import numpy as np
import math

class TemperatureScheduler:
    """
    Various temperature scheduling methods for contrastive learning
    """
    
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
        Cosine annealing with warm restarts (SGDR-style)
        Periodically restarts the schedule with increasing periods
        
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

    @staticmethod
    def get_scheduler(method_name):
        """
        Factory method to get scheduler function by name
        """
        schedulers = {
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

# Test the schedulers
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    epochs = np.arange(0, 20)
    methods = ['cosine_annealing', 'linear_decay', 'exponential_decay', 'step_decay', 
              'polynomial_decay', 'sigmoid_decay', 'warmup_cosine', 'constant', 'cyclic', 'no_scheduling',
              'adaptive_cosine_annealing', 'multi_cycle_cosine', 'cosine_with_restarts']
    
    plt.figure(figsize=(15, 10))
    for i, method in enumerate(methods):
        scheduler = TemperatureScheduler.get_scheduler(method)
        temps = [scheduler(epoch) for epoch in epochs]
        
        plt.subplot(4, 3, i+1)
        plt.plot(epochs, temps, 'b-', linewidth=2)
        plt.title(f'{method.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Temperature (τ)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('temperature_schedulers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
