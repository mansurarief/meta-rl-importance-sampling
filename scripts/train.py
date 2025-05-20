import os
import sys
import yaml
import torch
import numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.ceis import CEIS, CEISConfig
from src.utils.data_loader import DatasetLoader, DatasetConfig
from src.models.ego_policy import EgoPolicy
from src.models.social_policy import SocialPolicy
from src.environments.intersection import IntersectionEnv
from src.environments.roundabout import RoundaboutEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CEIS model on driving scenarios')
    parser.add_argument('--config', type=str, default='configs/ceis_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--scenario_id', type=str, default=None,
                      help='Specific scenario ID to train on')
    parser.add_argument('--scenario_type', type=str, choices=['intersection', 'roundabout'],
                      default=None, help='Type of scenario to train on')
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Directory containing the dataset')
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_environment(config: Dict, scenario_params: Dict):
    """Create environment based on scenario type."""
    if config['scenario_type'] == 'intersection':
        return IntersectionEnv(scenario_params)
    else:  # roundabout
        return RoundaboutEnv(scenario_params)

def train_ceis(
    config: Dict,
    data_loader: DatasetLoader,
    environment,
    device: torch.device
) -> Tuple[EgoPolicy, List[Dict]]:
    """Train using Cross Entropy Importance Sampling (CEIS) algorithm.
    
    This function implements the training loop for the CEIS algorithm. It:
    1. Initializes the ego and social policies
    2. Sets up the CEIS algorithm with the specified configuration
    3. Estimates the naturalistic distribution from the dataset
    4. Runs the training loop for the specified number of iterations
    5. Updates the mixture model periodically
    
    Args:
        config (Dict): Configuration dictionary containing training parameters
        data_loader (DatasetLoader): Dataset loader for accessing training data
        environment: The training environment (IntersectionEnv or RoundaboutEnv)
        device (torch.device): Device to run the training on (CPU/GPU)
    
    Returns:
        Tuple[EgoPolicy, List[Dict]]: Trained ego policy and training metrics history
    """
    # Initialize policies
    ego_policy = EgoPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    social_policy = SocialPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Initialize CEIS with configuration
    ceis_config = CEISConfig(
        num_iterations=config['num_iterations'],
        num_samples_per_iter=config['num_samples_per_iter'],
        elite_fraction=config['elite_fraction'],
        min_std=config['min_std'],
        initial_std=config['initial_std'],
        mixture_components=config['mixture_components']  # Number of Gaussian components (k)
    )
    ceis = CEIS(ceis_config)
    
    # Get naturalistic distribution from dataset
    naturalistic_dist = data_loader.estimate_naturalistic_distribution()
    
    # Initialize CEIS distribution with naturalistic parameters
    ceis.initialize_distribution(
        mean=naturalistic_dist['mean'],
        std=naturalistic_dist['std']
    )
    
    # Training loop
    metrics_history = []
    for iteration in tqdm(range(config['num_iterations']), desc="Training"):
        # Train one iteration of CEIS
        success_rate, collision_rate, timeout_rate = ceis.train_iteration(
            ego_policy,
            environment,
            naturalistic_dist
        )
        
        # Record metrics for this iteration
        metrics = {
            'iteration': iteration,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate
        }
        metrics_history.append(metrics)
        
        # Update mixture model periodically
        if iteration > 0 and iteration % config['mixture_update_freq'] == 0:
            gmm = ceis.create_mixture_model()
            # Update environment with new mixture model
            environment.update_social_policy_distribution(gmm)
    
    return ego_policy, metrics_history

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.scenario_id:
        config['scenario_id'] = args.scenario_id
    if args.scenario_type:
        config['scenario_type'] = args.scenario_type
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    # Validate scenario configuration
    if config['scenario_type'] == 'intersection':
        valid_scenarios = ['heckstrasse', 'bendplatz', 'frankenberg', 'neuweiler']
        if config['scenario_id'] not in valid_scenarios:
            raise ValueError(f"Invalid intersection scenario. Must be one of {valid_scenarios}")
    elif config['scenario_type'] == 'roundabout':
        valid_scenarios = ['round1', 'round2', 'round3', 'round4']
        if config['scenario_id'] not in valid_scenarios:
            raise ValueError(f"Invalid roundabout scenario. Must be one of {valid_scenarios}")
    else:
        raise ValueError("Invalid scenario type. Must be 'intersection' or 'roundabout'")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader
    data_config = DatasetConfig(
        data_dir=config['data_dir'],
        scenario_type=config['scenario_type'],
        scenario_id=config['scenario_id']
    )
    data_loader = DatasetLoader(data_config)
    
    # Load dataset
    if config['scenario_type'] == 'intersection':
        data_loader.load_ind_dataset()
    else:
        data_loader.load_roundd_dataset()
    
    # Get scenario parameters
    scenario_params = data_loader.get_scenario_parameters()
    
    # Create environment
    environment = create_environment(config, scenario_params)
    
    # Train
    ego_policy, metrics_history = train_ceis(
        config,
        data_loader,
        environment,
        device
    )
    
    # Save results
    output_dir = os.path.join(config['output_dir'], 
                            f"{config['scenario_type']}_{config['scenario_id']}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save policy
    torch.save(ego_policy.state_dict(), os.path.join(output_dir, 'ego_policy.pt'))
    
    # Save metrics
    import json
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)

if __name__ == '__main__':
    main() 