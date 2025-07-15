import gymnasium as gym
import torch
from dqn import DQNAgent, set_seed, evaluate_policy
from wrapper import FlatObsImageOnlyWrapper
from minigrid.wrappers import FlatObsWrapper

# 1. Umgebung mit Render-Modus erstellen
env = gym.make("MiniGrid-Empty-8x8-v0")
env = FlatObsImageOnlyWrapper(env)
env = gym.wrappers.TimeLimit(env, max_episode_steps=2000)  # z.B. 1000 Schritte
set_seed(env, 10)

# 2. Agent initialisieren (Parameter wie beim Training!)
agent = DQNAgent(
    env=env,
    seed=1,
    buffer_capacity=50000,
    batch_size=32,
    useNoisyNet=True,
)

# 3. Modell laden
checkpoint = torch.load("models/dqn_trained_model_MiniGrid_Empty_5x5_v0_20250715_162459.pth")
agent.q.load_state_dict(checkpoint["parameters"])
agent.optimizer.load_state_dict(checkpoint["optimizer"])

# 4. Simulation (eine Episode)
score = evaluate_policy(env, agent, turns=50)
print(score)

env.close()
