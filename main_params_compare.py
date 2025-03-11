import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime
import time
from scipy import stats
from main_conv_compare import run_experiment

class LearningComparison:
    def __init__(self, individual_data: List[np.ndarray], federated_data: List[np.ndarray]):
        self.individual_data = np.array(individual_data)
        self.federated_data = np.array(federated_data)
        self.metrics = {}
        
    def analyze_communication_overhead(self, sync_interval: int, model_size_bytes: int) -> Dict:
        """Calculate communication costs for both approaches"""
        n_episodes = self.individual_data.shape[1]
        n_agents = len(self.individual_data)
        
        # Individual learning only needs initial model distribution
        individual_comm = model_size_bytes * n_agents
        
        # Federated learning needs periodic model synchronization
        n_syncs = n_episodes // sync_interval
        federated_comm = model_size_bytes * n_agents * n_syncs
        
        return {
            'individual_communication': individual_comm,
            'federated_communication': federated_comm,
            'communication_ratio': federated_comm / individual_comm
        }
    
    def analyze_stability(self, window_size: int = 10) -> Dict:
        """Analyze learning stability using reward variance"""
        # Calculate rolling variance
        def get_rolling_variance(data):
            return pd.DataFrame(data).rolling(window=window_size).var().mean().mean()
        
        individual_variance = get_rolling_variance(self.individual_data)
        federated_variance = get_rolling_variance(self.federated_data)
        
        # Calculate sudden performance drops
        def count_performance_drops(data, threshold=0.2):
            drops = 0
            for trial in data:
                drops += np.sum(np.diff(trial) < -threshold)
            return drops / len(data)
        
        return {
            'individual_variance': individual_variance,
            'federated_variance': federated_variance,
            'individual_drops': count_performance_drops(self.individual_data),
            'federated_drops': count_performance_drops(self.federated_data)
        }
    
    def analyze_resource_utilization(self, 
                                   computation_units: List[int],
                                   velocities: List[int]) -> Dict:
        """Analyze how efficiently each approach uses available resources"""
        # Calculate resource utilization efficiency
        def calculate_efficiency(rewards, units, velocities):
            normalized_rewards = rewards / np.max(rewards)
            resource_usage = np.sum(units) * np.mean(velocities)
            return np.mean(normalized_rewards) / resource_usage
        
        individual_efficiency = calculate_efficiency(
            self.individual_data, computation_units, velocities)
        federated_efficiency = calculate_efficiency(
            self.federated_data, computation_units, velocities)
        
        return {
            'individual_efficiency': individual_efficiency,
            'federated_efficiency': federated_efficiency,
            'efficiency_ratio': federated_efficiency / individual_efficiency
        }
    
    def analyze_adaptation(self, 
                         channel_patterns: List[str],
                         episode_split: int = None) -> Dict:
        """Analyze how well each approach adapts to different conditions"""
        if episode_split is None:
            episode_split = self.individual_data.shape[1] // 2
            
        # Calculate performance improvement rate
        def calculate_improvement(data):
            first_half = data[:, :episode_split]
            second_half = data[:, episode_split:]
            return (np.mean(second_half) - np.mean(first_half)) / np.mean(first_half)
        
        # Calculate per-pattern performance
        def per_pattern_performance(data, patterns):
            pattern_perf = {}
            unique_patterns = set(patterns)
            for pattern in unique_patterns:
                indices = [i for i, p in enumerate(patterns) if p == pattern]
                pattern_perf[pattern] = np.mean(data[indices])
            return pattern_perf
        
        return {
            'individual_improvement': calculate_improvement(self.individual_data),
            'federated_improvement': calculate_improvement(self.federated_data),
            'individual_pattern_perf': per_pattern_performance(
                self.individual_data, channel_patterns),
            'federated_pattern_perf': per_pattern_performance(
                self.federated_data, channel_patterns)
        }
    
    def analyze_robustness(self, noise_level: float = 0.1) -> Dict:
        """Analyze robustness to noise and perturbations"""
        # Add noise to rewards and check performance degradation
        def calculate_robustness(data, noise_std):
            noisy_data = data + np.random.normal(0, noise_std, data.shape)
            return np.mean(np.abs(data - noisy_data)) / np.mean(data)
        
        return {
            'individual_robustness': calculate_robustness(
                self.individual_data, noise_level),
            'federated_robustness': calculate_robustness(
                self.federated_data, noise_level)
        }
    
    def analyze_statistical_significance(self) -> Dict:
        """Perform statistical tests to compare approaches"""
        # Perform t-test on final performance
        t_stat, p_value = stats.ttest_ind(
            self.individual_data[:, -1],
            self.federated_data[:, -1]
        )
        
        # Calculate effect size (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_se
        
        effect_size = cohens_d(
            self.individual_data[:, -1],
            self.federated_data[:, -1]
        )
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def plot_comprehensive_comparison(self, save_path: str = None):
        """Create a comprehensive visualization of the comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Learning curves with confidence intervals
        ax = axes[0, 0]
        episodes = range(self.individual_data.shape[1])
        
        def plot_with_ci(data, label, color):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ax.plot(episodes, mean, label=label, color=color)
            ax.fill_between(episodes, mean - std, mean + std, 
                          alpha=0.2, color=color)
        
        plot_with_ci(self.individual_data, 'Individual', 'blue')
        plot_with_ci(self.federated_data, 'Federated', 'red')
        ax.set_title('Learning Curves')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Reward')
        ax.legend()
        
        # 2. Performance distribution violin plot
        ax = axes[0, 1]
        final_individual = self.individual_data[:, -1]
        final_federated = self.federated_data[:, -1]
        
        data = [final_individual, final_federated]
        violin = ax.violinplot(data)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Individual', 'Federated'])
        ax.set_title('Final Performance Distribution')
        
        # 3. Learning stability
        ax = axes[1, 0]
        window = 10
        individual_smooth = pd.DataFrame(self.individual_data).rolling(window).mean()
        federated_smooth = pd.DataFrame(self.federated_data).rolling(window).mean()
        
        ax.plot(individual_smooth.mean(), label='Individual', color='blue')
        ax.plot(federated_smooth.mean(), label='Federated', color='red')
        ax.set_title('Smoothed Learning Progress')
        ax.legend()
        
        # 4. Performance improvement rate
        ax = axes[1, 1]
        episode_split = self.individual_data.shape[1] // 2
        
        def improvement_data(data):
            first_half = np.mean(data[:, :episode_split], axis=1)
            second_half = np.mean(data[:, episode_split:], axis=1)
            return (second_half - first_half) / first_half
        
        individual_imp = improvement_data(self.individual_data)
        federated_imp = improvement_data(self.federated_data)
        
        ax.bar([1, 2], [np.mean(individual_imp), np.mean(federated_imp)],
               yerr=[np.std(individual_imp), np.std(federated_imp)],
               capsize=5, color=['blue', 'red'])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Individual', 'Federated'])
        ax.set_title('Performance Improvement Rate')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

def main():
    # Example usage with your existing experiment data
    individual_rewards, federated_rewards = run_experiment(
        n_trials=5,
        n_agents=5,
        n_episodes=1000,
        sync_interval=100
    )
    
    # Initialize comparison analyzer
    comparison = LearningComparison(individual_rewards, federated_rewards)
    
    # Run all analyses
    communication_metrics = comparison.analyze_communication_overhead(
        sync_interval=10,
        model_size_bytes=1000000  # Example model size
    )
    
    stability_metrics = comparison.analyze_stability()
    
    resource_metrics = comparison.analyze_resource_utilization(
        computation_units=[10, 50, 50, 100, 100],
        velocities=[10, 20, 20, 30, 30]
    )
    
    adaptation_metrics = comparison.analyze_adaptation(
        channel_patterns=['urban', 'suburban', 'suburban', 'rural', 'rural']
    )
    
    robustness_metrics = comparison.analyze_robustness()
    
    statistical_metrics = comparison.analyze_statistical_significance()
    
    # Generate visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison.plot_comprehensive_comparison(
        save_path=f'comprehensive_comparison_{timestamp}.png'
    )
    
    # Print results
    print("\nComprehensive Comparison Results:")
    print("\nCommunication Overhead:")
    print(f"Individual: {communication_metrics['individual_communication']:,} bytes")
    print(f"Federated: {communication_metrics['federated_communication']:,} bytes")
    print(f"Ratio: {communication_metrics['communication_ratio']:.2f}x")
    
    print("\nLearning Stability:")
    print(f"Individual Variance: {stability_metrics['individual_variance']:.4f}")
    print(f"Federated Variance: {stability_metrics['federated_variance']:.4f}")
    
    print("\nResource Utilization Efficiency:")
    print(f"Individual: {resource_metrics['individual_efficiency']:.4f}")
    print(f"Federated: {resource_metrics['federated_efficiency']:.4f}")
    
    print("\nAdaptation Rate:")
    print(f"Individual Improvement: {adaptation_metrics['individual_improvement']:.2%}")
    print(f"Federated Improvement: {adaptation_metrics['federated_improvement']:.2%}")
    
    print("\nRobustness to Noise:")
    print(f"Individual: {robustness_metrics['individual_robustness']:.4f}")
    print(f"Federated: {robustness_metrics['federated_robustness']:.4f}")
    
    print("\nStatistical Significance:")
    print(f"p-value: {statistical_metrics['p_value']:.4f}")
    print(f"Effect Size: {statistical_metrics['effect_size']:.4f}")

if __name__ == "__main__":
    main()