import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture

@dataclass
class CEISConfig:
    """Configuration for Cross Entropy Importance Sampling (CEIS) algorithm.
    
    Parameters:
        num_iterations (int): Number of training iterations
        num_samples_per_iter (int): Number of samples generated per iteration
        elite_fraction (float): Fraction of elite samples used for updating the distribution
        min_std (float): Minimum standard deviation to prevent collapse
        initial_std (float): Initial standard deviation for the distribution
        mixture_components (int): Number of Gaussian components (k) in the mixture model
    """
    num_iterations: int = 10
    num_samples_per_iter: int = 1000
    elite_fraction: float = 0.1
    min_std: float = 0.1
    initial_std: float = 0.5
    mixture_components: int = 3

class CEIS:
    """Cross Entropy Importance Sampling (CEIS) algorithm implementation.
    
    This class implements the CEIS algorithm for training policies in interactive environments.
    It uses a mixture of Gaussians to model the sampling distribution and updates it iteratively
    based on the performance of sampled trajectories.
    
    The algorithm works as follows:
    1. Initialize a sampling distribution (mixture of Gaussians)
    2. For each iteration:
        a. Sample trajectories from the current distribution
        b. Evaluate trajectories using the policy
        c. Select elite samples based on performance
        d. Update the distribution parameters using elite samples
        e. Optionally update the mixture model
    """
    
    def __init__(self, config: CEISConfig):
        """Initialize CEIS algorithm with configuration.
        
        Args:
            config (CEISConfig): Configuration parameters for the algorithm
        """
        self.config = config
        self.distribution = None
        self.mixture_model = None
        self.history = []
        
    def initialize_distribution(self, mean: float, std: float) -> None:
        """Initialize the sampling distribution."""
        self.distribution = {
            'mean': mean,
            'std': std
        }
        
    def sample_beta(self, num_samples: int) -> np.ndarray:
        """Sample beta values from current distribution."""
        if self.distribution is None:
            raise ValueError("Distribution not initialized")
            
        return np.random.normal(
            self.distribution['mean'],
            self.distribution['std'],
            num_samples
        )
        
    def update_distribution(self, samples: np.ndarray, rewards: np.ndarray) -> None:
        """Update the sampling distribution using elite samples."""
        if len(samples) == 0:
            return
            
        # Sort samples by reward
        sorted_indices = np.argsort(rewards)
        elite_size = max(1, int(len(samples) * self.config.elite_fraction))
        elite_indices = sorted_indices[:elite_size]
        elite_samples = samples[elite_indices]
        
        # Update distribution parameters
        new_mean = np.mean(elite_samples)
        new_std = max(self.config.min_std, np.std(elite_samples))
        
        self.distribution = {
            'mean': new_mean,
            'std': new_std
        }
        
        self.history.append(self.distribution.copy())
        
    def create_mixture_model(self) -> GaussianMixture:
        """Create a Gaussian Mixture Model from history."""
        if not self.history:
            raise ValueError("No history available for mixture model")
            
        means = np.array([d['mean'] for d in self.history])
        covars = np.array([d['std']**2 for d in self.history])
        
        gmm = GaussianMixture(
            n_components=min(self.config.mixture_components, len(self.history)),
            covariance_type='spherical'
        )
        
        # Reshape for GMM
        X = means.reshape(-1, 1)
        gmm.fit(X)
        
        self.mixture_model = gmm
        
        return gmm
        
    def compute_importance_weights(
        self,
        samples: np.ndarray,
        naturalistic_dist: Dict[str, float]
    ) -> np.ndarray:
        """Compute importance weights for samples."""
        if self.distribution is None:
            raise ValueError("Distribution not initialized")
            
        # Compute proposal density
        proposal_density = np.exp(
            -0.5 * ((samples - self.distribution['mean']) / 
                    self.distribution['std'])**2
        ) / (self.distribution['std'] * np.sqrt(2 * np.pi))
        
        # Compute naturalistic density
        naturalistic_density = np.exp(
            -0.5 * ((samples - naturalistic_dist['mean']) / 
                    naturalistic_dist['std'])**2
        ) / (naturalistic_dist['std'] * np.sqrt(2 * np.pi))
        
        # Compute importance weights
        weights = naturalistic_density / proposal_density
        return weights / np.mean(weights)  # Normalize weights
        
    def train_iteration(
        self,
        policy,
        environment,
        naturalistic_dist: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """Perform one training iteration."""
        # Sample scenarios
        betas = self.sample_beta(self.config.num_samples_per_iter)
        
        # Evaluate scenarios
        rewards = []
        for beta in betas:
            reward = environment.evaluate_policy(policy, beta)
            rewards.append(reward)
        rewards = np.array(rewards)
        
        # Compute importance weights
        weights = self.compute_importance_weights(betas, naturalistic_dist)
        
        # Update policy using weighted rewards
        policy.update(betas, rewards, weights)
        
        # Update sampling distribution
        self.update_distribution(betas, rewards)
        
        # Compute metrics
        success_rate = np.mean(rewards > 0)
        collision_rate = np.mean(rewards < -1)
        timeout_rate = np.mean(rewards == 0)
        
        return success_rate, collision_rate, timeout_rate 