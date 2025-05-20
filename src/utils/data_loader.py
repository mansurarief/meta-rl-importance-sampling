import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.neighbors import KernelDensity

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    data_dir: str
    scenario_type: str  # 'intersection' or 'roundabout'
    scenario_id: str    # e.g., 'heckstrasse' for InD
    min_trajectory_length: int = 10
    max_trajectory_length: int = 100
    time_step: float = 0.1  # seconds

class DatasetLoader:
    """Loader for InD and RoundD datasets."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = None
        self.naturalistic_dist = None
        
    def load_ind_dataset(self) -> pd.DataFrame:
        """Load InD dataset for T-intersections."""
        if self.config.scenario_type != 'intersection':
            raise ValueError("InD dataset only supports intersection scenarios")
            
        # Load trajectory data
        traj_file = os.path.join(
            self.config.data_dir,
            'ind',
            f'{self.config.scenario_id}_trajectories.csv'
        )
        
        if not os.path.exists(traj_file):
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
            
        # Load and preprocess data
        df = pd.read_csv(traj_file)
        
        # Filter by scenario type and length
        df = df[
            (df['scenario_type'] == self.config.scenario_type) &
            (df['trajectory_length'] >= self.config.min_trajectory_length) &
            (df['trajectory_length'] <= self.config.max_trajectory_length)
        ]
        
        self.data = df
        return df
        
    def load_roundd_dataset(self) -> pd.DataFrame:
        """Load RoundD dataset for roundabouts."""
        if self.config.scenario_type != 'roundabout':
            raise ValueError("RoundD dataset only supports roundabout scenarios")
            
        # Load trajectory data
        traj_file = os.path.join(
            self.config.data_dir,
            'roundd',
            f'{self.config.scenario_id}_trajectories.csv'
        )
        
        if not os.path.exists(traj_file):
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
            
        # Load and preprocess data
        df = pd.read_csv(traj_file)
        
        # Filter by scenario type and length
        df = df[
            (df['scenario_type'] == self.config.scenario_type) &
            (df['trajectory_length'] >= self.config.min_trajectory_length) &
            (df['trajectory_length'] <= self.config.max_trajectory_length)
        ]
        
        self.data = df
        return df
        
    def estimate_naturalistic_distribution(self) -> Dict[str, float]:
        """Estimate naturalistic distribution of beta values."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_ind_dataset or load_roundd_dataset first.")
            
        # Extract beta values from trajectories
        beta_values = self.data['beta'].values
        
        # Fit kernel density estimator
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
        kde.fit(beta_values.reshape(-1, 1))
        
        # Compute mean and std
        mean = np.mean(beta_values)
        std = np.std(beta_values)
        
        self.naturalistic_dist = {
            'mean': float(mean),
            'std': float(std),
            'kde': kde
        }
        
        return self.naturalistic_dist
        
    def get_training_data(self, split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_ind_dataset or load_roundd_dataset first.")
            
        # Shuffle data
        df = self.data.sample(frac=1, random_state=42)
        
        # Split
        train_size = int(len(df) * split)
        train_data = df[:train_size]
        val_data = df[train_size:]
        
        return train_data, val_data
        
    def get_scenario_parameters(self) -> Dict[str, float]:
        """Get scenario-specific parameters."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_ind_dataset or load_roundd_dataset first.")
            
        if self.config.scenario_type == 'intersection':
            return {
                'intersection_width': float(self.data['intersection_width'].iloc[0]),
                'intersection_length': float(self.data['intersection_length'].iloc[0]),
                'approach_length': float(self.data['approach_length'].iloc[0])
            }
        else:  # roundabout
            return {
                'roundabout_radius': float(self.data['roundabout_radius'].iloc[0]),
                'approach_length': float(self.data['approach_length'].iloc[0]),
                'num_arms': int(self.data['num_arms'].iloc[0])
            } 