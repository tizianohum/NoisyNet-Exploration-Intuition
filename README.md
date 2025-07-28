# NoisyNet Exploration Intuition

A Deep Q-Learning implementation with NoisyNet exploration for reinforcement learning environments.

## Overview

This project implements and compares different exploration strategies in reinforcement learning, with a focus on NoisyNet versus epsilon-greedy exploration in MiniGrid environments.

## Features

- **DQN Agent** with configurable exploration strategies
- **NoisyNet implementation** for parameter space exploration
- **MiniGrid environment support** with custom wrappers
- **Comprehensive experiment tracking** with automatic data saving
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
└── notebooks/             # Jupyter notebooks for analysis
    └── test.ipynb
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
  name: MiniGrid-Empty-5x5-v0

agent:
  buffer_capacity: 50000
  batch_size: 32
  learning_rate: 0.001
  gamma: 0.99
  use_noisy_net: true  # Set to false for epsilon-greedy

train:
  num_frames: 100000
  eval_interval: 1000
```

### Testing Trained Models

Test a specific experiment:

```bash
python testMMDL.py
```

Update the timestamp in the script to match your experiment:

```python
timestamp = "20250728_133214"  # Your experiment timestamp
```

### Analysis

Use the analysis tools to visualize results:

```python
from auswertung import Auswerter

# Load experiment data
analyzer = Auswerter("20250728_133214")

# Generate training plots
analyzer.simpleplot(save=True)

# Create heatmaps (for grid environments)
analyzer.create_heatmap()
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

## Key Components

### NoisyNet Implementation

- **Factorized Gaussian noise** for efficient parameter space exploration
- **Automatic noise scheduling** without manual epsilon decay
- **Deterministic evaluation mode** for consistent testing

### Environment Wrappers

- **FlatObsWrapper**: Flattens MiniGrid observations for DQN
- **FlatObsImageOnlyWrapper**: Image-only observations
- **Custom reward shaping** and observation preprocessing

### Experiment Tracking

- **Automatic timestamping** of all experiments
- **Complete configuration backup** for reproducibility
- **Training metrics logging** (rewards, loss, epsilon values)
- **Agent position tracking** for exploration analysis

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
