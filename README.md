# NoisyNet Exploration Intuition

A Deep Q-Learning implementation with NoisyNet exploration for reinforcement learning environments.

## Overview

This project implements and compares different exploration strategies in reinforcement learning, with a focus on NoisyNet versus epsilon-greedy exploration in MiniGrid and Cart-Pole environments.

## Features

- **DQN Agent** with configurable exploration strategies
- **NoisyNet implementation** for exploration
- **Experiment tracking** with automatic data saving
- **Visualization tools** for training analysis and heatmap generation
- **Hydra configuration management** for reproducible experiments

## Setup

### Prerequisites

- Python 3.11+
- UV package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NoisyNet-Exploration-Intuition
   ```

2. **Create virtual environment**
   ```bash
   uv venv .noisynet --python 3.11
   ```

3. **Activate environment**
   ```bash
   # macOS/Linux
   source .noisynet/bin/activate
   
   # Windows
   .noisynet\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

## Project Structure

```
NoisyNet-Exploration-Intuition/
├── README.md
├── requirements.txt
├── dqn.py                    # Main DQN agent implementation
├── networks.py               # Neural network architectures (Q-Network, NoisyNet)
├── buffers.py               # Experience replay buffer
├── abstract_agent.py        # Base agent interface
├── wrapper.py               # Environment wrappers
├── testMMDL.py             # Model testing script
├── auswertung.py           # Results analysis tools
├── configs/
│   └── agent/
│       └── config.yaml     # Hyperparameter configuration
├── RAW_Data/               # Experiment results (auto-generated)
│   └── YYYYMMDD_HHMMSS/   # Timestamped experiment folders
│       ├── models/        # Trained model checkpoints
│       ├── training_data/ # Training metrics and logs
│       ├── heatmap_data/  # Agent position data for visualization
│       └── config.yaml    # Experiment configuration backup
└── test.ipynb             # notebook for quick tests
```

## Usage

### Training

Run a training experiment:

```bash
python dqn.py
```

Configure parameters in `configs/agent/config.yaml`:

```yaml
env:
  name: CartPole-v1  # Gym environment name

seed: 0

agent:
  buffer_capacity:    100000    # max replay buffer size
  batch_size:         256       # minibatch size
  learning_rate:      0.0001    # maps to DQNAgent’s lr
  gamma:              0.99
  epsilon_start:      1.0
  epsilon_final:      0.01
  epsilon_decay:      5000
  target_update_freq: 1000
  use_noisy_net:      True     # Set to True to use NoisyNet exploration
  minReward:          0
  maxReward:          500

train:
  num_frames:     200000   # total env steps
  eval_interval:  1000    # print avg reward every this many episodes
```

### Hyperparameter Sweeping

Use Hyperparameter Sweeping, to find a good Hyperparameter Configuration for the different environments and algorithms. For a detailed explanation visit: https://github.com/automl/hypersweeper

For the sweeping a config_sweeper.yaml is used:

```yaml
# configs/agent/config_sweeper.yaml
defaults:
  - override hydra/sweeper: optuna

env:
  name: CartPole-v1  # Gym environment name

seed: 0

agent:
  buffer_capacity: 50000  # Default Wert
  batch_size: 256
  learning_rate: 0.001
  gamma: 0.95
  epsilon_start: 1.0
  epsilon_final: 0.01
  epsilon_decay: 10000
  target_update_freq: 1000
  use_noisy_net: False
  minReward:          -1
  maxReward:          1

train:
  num_frames:   50000 # Kürzer für schnelle Tests
  eval_interval: 1000
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: ${seed}
    direction: maximize
    study_name: buffer_capacity_sweep_v3
    n_trials: 20
    params:
      agent.buffer_capacity: choice(10000, 25000, 50000)
      agent.batch_size: choice(64, 128, 256)
      agent.learning_rate: choice(0.0001, 0.005, 0.001)
      agent.gamma: choice(0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99)
      agent.epsilon_decay: choice(5000, 10000, 20000)
      agent.target_update_freq: choice(500, 1000, 5000)
```

run with:
````
python3 dqn.py --multirun
````

### Testing Trained Models

Test a specific experiment:

```bash
python testMMDL.py
```

Update the timestamp in the script to match your experiment (The timestamp is also the foldername, where your data was saved). Make sure your experiment has the same .yaml as your main .yaml in configs/agent. The file is saved in your experiment folder. Copy the parameters or change the path in your testMMDL.py.

```python
timestamp = "20250728_133214"  # Your experiment timestamp
```

### Analysis

Use the analysis tools to visualize results:

```python
from auswertung import Auswerter

# Load experiment data
algorithm_timestamps = {"DQN-NoisyNet": timestamp}
analyzer = Auswerter(algorithm_timestamps)

# Generate training plots
analyzer.simpleplot(save=True)

# Create heatmaps (for grid environments)
auswertung.heatmap(8,8) # for 8x8 minigrid environment
```

## Configuration

### Environment Settings

```yaml
env:
  name: MiniGrid-Empty-5x5-v0  # Environment name
```

### Agent Hyperparameters

```yaml
agent:
  buffer_capacity: 50000      # Replay buffer size
  batch_size: 32             # Mini-batch size
  learning_rate: 0.001       # Learning rate
  gamma: 0.99               # Discount factor
  epsilon_start: 1.0        # Initial epsilon (if not using NoisyNet)
  epsilon_final: 0.01       # Final epsilon
  epsilon_decay: 50000      # Epsilon decay steps
  target_update_freq: 1000  # Target network update frequency
  use_noisy_net: true       # Use NoisyNet vs epsilon-greedy
```

### Training Settings

```yaml
train:
  num_frames: 100000        # Total training steps
  eval_interval: 1000       # Evaluation frequency
```


## Results

Each experiment automatically saves:

- **Model checkpoints** (`models/dqn_trained_model_*.pth`)
- **Training data** (`training_data/dqn_training_data_*.csv`)
- **Exploration heatmaps** (`heatmap_data/positions_*.csv`)
- **Configuration backup** (`config.yaml`)

## Dependencies

Main dependencies include:

```
torch>=2.0.0
gymnasium[box2d]>=0.29.0
minigrid>=2.3.0
hydra-core>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
```

See `requirements.txt` for the complete list.

## Adding New Packages

To add new dependencies:

```bash
uv pip install <package-name>
uv pip freeze > requirements.txt
```
