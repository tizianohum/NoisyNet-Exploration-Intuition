# Important Hyperparametersweep-Results and Timestamps

## Sweep Parameters

- Learning Rate
- Gamma
- Epsilon Decay
- Batch Size
- Replay Buffer
- Target Update Frequency

## List Environments

- Minigrid 5x5
- Minigrid 8x8
- CartPole-v1

# Hyperparameters Environment

8x8 epsilon greedy
name: optuna
best_params:
  agent.gamma: 0.95
  agent.learning_rate: 0.001
  agent.buffer_capacity: 25000
  agent.batch_size: 256
  agent.target_update_freq: 1000
  agent.epsilon_decay: 10000
best_value: 0.941640625

Noisy:
Parameter 8x8:
agent:
  buffer_capacity: 75000  # Default Wert
  batch_size: 256
  learning_rate: 0.0001
  gamma: 0.9
  epsilon_start: 1.0
  epsilon_final: 0.05
  epsilon_decay: 20000
  target_update_freq: 1000
  use_noisy_net: True

train:
  num_frames: 50000  # Kürzer für schnelle Tests
  eval_interval: 1000

Params Lunar Lander epsilon greedy
name: optuna
best_params:
  agent.gamma: 0.95
  agent.learning_rate: 0.0001
  agent.buffer_capacity: 25000
  agent.target_update_freq: 1000
  agent.epsilon_decay: 5000
best_value: 15.114377614969973

5x5 sweep mit epsilon greedy
name: optuna
best_params:
  agent.gamma: 0.6
  agent.learning_rate: 0.0001
  agent.buffer_capacity: 10000
  agent.batch_size: 64
  agent.target_update_freq: 1000
  agent.epsilon_decay: 5000
best_value: 0.9360999999999999

5x5 noisy
name: optuna
best_params:
  agent.gamma: 0.7
  agent.batch_size: 256
  agent.learning_rate: 0.0001
  agent.buffer_capacity: 25000
  agent.target_update_freq: 1000
best_value: 0.9117999999999998

5x5 obsticle epsilon greedy
name: optuna
best_params:
  agent.gamma: 0.9
  agent.learning_rate: 0.001
  agent.epsilon_decay: 5000
best_value: 0.9145

5x5 obsticle noise k=4
name: optuna
best_params:
  agent.gamma: 0.8
  agent.learning_rate: 0.001
best_value: 0.7884



# Important Timestamps

## 5x5 Empty
30 rund 5x5 noisy, noisy params: 20250802_104744
30 runs 5x5 epsilon greedy, noisy params: 20250802_112343
30 runs 5x5 k=4: 20250803_123825

## 8x8 Empty 30 runs:
noise reduction: 20250804_231747
noise: 20250805_025033
greedy: 20250805_062148

## 8x8 Empty 30 runs with goal tile in heatmap:
noise reduction: 20250805_153009
noise: 20250806_091853
greedy: 20250806_125218

## Cartpole-v1
15 runs, cartpole greedy: 20250804_004134
15 runs, cartpole noisy, noise reduction off: 20250804_013949
15 runs, cartpole noisy, noise reduction on(wrong learning rate): 20250805_111301

30 runs, cartpole greedy: 20250807_064010
15 runs, cartpole noisy, noise reduction on: 20250807_150346

## Old
30 runs 8x8 epsilon greedy, greedy params: 20250801_223311
30 runs 8x8 noisy:  20250801_234458
30 runs 8x8 k = 0.1, noisy: 20250802_235544
20 runs 8x8 k = 4, noisy: 20250803_101016

30 rund 5x5 noisy, noisy params: 20250802_104744
30 runs 5x5 epsilon greedy, noisy params: 20250802_112343
30 runs 5x5 k=4: 20250803_123825
30 runs 8x8 k = 0.1, noisy: 20250802_235544
20 runs 8x8 k = 4, noisy: 20250803_101016
30 runs 5x5 k=0.1: 20250803_131905
30 runs 5x5 k=1: 20250803_135322
15 runs, cartpole noisy, noise reduction on(wrong rewards min max): 20250804_043455
30 run 5x5 obstacle greedy: 20250803_160210  (obstacle was to hard to solve for noisy)
