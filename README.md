# Meta-RL with Importance Sampling for Interactive Environments

[![arXiv](https://img.shields.io/badge/arXiv-2407.15839-b31b1b.svg)](https://arxiv.org/abs/2407.15839)

This repository contains the implementation of the paper "Importance Sampling-Guided Meta-Training for Intelligent Agents in Highly Interactive Environments" (IEEE Robotics and Automation Letters, 2024) by Mansur Arief, Mike Timmerman, Jiachen Li, David Isele, and Mykel Kochenderfer.

## Overview

This project implements a novel training framework that integrates guided meta reinforcement learning with importance sampling (IS) to optimize training distributions for navigating highly interactive driving scenarios. The framework is particularly effective for scenarios like T-intersections and roundabouts, where traditional methods may struggle with balancing common and extreme cases.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/meta-rl-importance-sampling.git
cd meta-rl-importance-sampling

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
meta-rl-importance-sampling/
├── baselines/               # Baseline implementations
├── beta_dist/              # Beta distribution utilities
├── configs/                # Configuration files
│   └── driving_gpu.yaml   # Main configuration file
├── data/                   # Dataset storage
├── driving_sim/           # Driving simulation environment
├── experiments/           # Experiment results and logs
├── IS_normalizing_flows/  # Importance sampling with normalizing flows
├── pretext/              # Pretext task implementations
├── rl/                   # Reinforcement learning components
├── scripts/              # Utility scripts
├── src/                  # Core source code
├── trained_models/       # Saved model checkpoints
├── batch_eval_ego.sh    # Batch evaluation script
├── batch_train_ego.sh   # Batch training script
├── ego_social_test.py   # Testing script for ego and social policies
├── latent_visualization.py # Visualization utilities
├── pretext_collect_data.py # Data collection for pretext tasks
├── pretext_train.py     # Pretext task training
├── train_ego_with_trained_social.py # Ego policy training
├── train_social_with_RLEgo.py # Social policy training
└── train_social_with_RLEgo_part2.py # Extended social policy training
```

## Usage

### Training

The project supports multiple training approaches:

1. Pretext Task Training:
```bash
python pretext_train.py
```

2. Social Policy Training:
```bash
python train_social_with_RLEgo.py
```

3. Ego Policy Training:
```bash
python train_ego_with_trained_social.py
```

4. Batch Training:
```bash
./batch_train_ego.sh
```

5. CEIS Training with Various Mixture Components:
```bash
# Run CEIS with a specific k value
python scripts/train.py configs/ceis_config.yaml

# Run CEIS for multiple k values (1, 2, 3, 5, 10)
for k in 1 2 3 5 10; do
  sed "s/^mixture_components:.*/mixture_components: $k/" configs/ceis_config.yaml > configs/ceis_config_k${k}.yaml
  python scripts/train.py configs/ceis_config_k${k}.yaml
done
```

The CEIS (Cross Entropy Importance Sampling) algorithm can be configured through `configs/ceis_config.yaml`. Key parameters include:
- `mixture_components`: Number of Gaussian components (k) in the mixture model
- `num_iterations`: Number of training iterations
- `num_samples_per_iter`: Number of samples per iteration
- `elite_fraction`: Fraction of elite samples used for updating the distribution
- `min_std`: Minimum standard deviation for the distribution
- `initial_std`: Initial standard deviation
- `mixture_update_freq`: Frequency of updating the mixture model

### Evaluation

To evaluate trained models:

```bash
# Single evaluation
python ego_social_test.py

# Batch evaluation
./batch_eval_ego.sh
```


## Configuration

The main configuration file is `configs/ceis_config.yaml`. This file contains settings for:
- Environment parameters
- Training hyperparameters
- Model architectures
- Data processing options

### Dataset Selection

The codebase supports two types of driving scenarios:
1. T-intersections (using InD dataset)
2. Roundabouts (using RoundD dataset)

To switch between scenarios, modify the following parameters in `configs/ceis_config.yaml`:

```yaml
# For T-intersection (InD dataset)
scenario_type: "intersection"
scenario_id: "heckstrasse"  # Options: "heckstrasse", "bendplatz", "frankenberg", "neuweiler"

# For Roundabout (RoundD dataset)
scenario_type: "roundabout"
scenario_id: "round1"  # Options: "round1", "round2", "round3", "round4"
```

You can also specify the dataset path:
```yaml
data_dir: "data/ind"  # For InD dataset
data_dir: "data/roundd"  # For RoundD dataset
```

### Running Different Scenarios

1. **T-intersection Training:**
```bash
# Using default config
python scripts/train.py configs/ceis_config.yaml

# With specific intersection
python scripts/train.py --config configs/ceis_config.yaml --scenario_id bendplatz
```

2. **Roundabout Training:**
```bash
# Using default config
python scripts/train.py configs/ceis_config.yaml

# With specific roundabout
python scripts/train.py --config configs/ceis_config.yaml --scenario_id round2
```

3. **Batch Training Multiple Scenarios:**
```bash
# Train on all T-intersections
for scenario in heckstrasse bendplatz frankenberg neuweiler; do
  python scripts/train.py --config configs/ceis_config.yaml --scenario_id $scenario
done

# Train on all roundabouts
for scenario in round1 round2 round3 round4; do
  python scripts/train.py --config configs/ceis_config.yaml --scenario_id $scenario
done
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{arief2024importance,
  title={Importance Sampling-Guided Meta-Training for Intelligent Agents in Highly Interactive Environments},
  author={Arief, Mansur and Timmerman, Mike and Li, Jiachen and Isele, David and Kochenderfer, Mykel J},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## Maintainer

This repository is maintained by Mansur Arief (mansur.arief@stanford.edu).
