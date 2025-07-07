import gymnasium as gym
import torch
from dqn import DQNAgent, set_seed

# 1. Umgebung mit Render-Modus erstellen
env = gym.make("CartPole-v1", render_mode="human")
env = gym.wrappers.TimeLimit(env, max_episode_steps=2000)  # z.B. 1000 Schritte
set_seed(env, 10)

# 2. Agent initialisieren (Parameter wie beim Training!)
agent = DQNAgent(
    env=env,
    seed=1,
    buffer_capacity=50000,
    batch_size=32,
)

# 3. Modell laden
checkpoint = torch.load("models/dqn_trained_model_CartPole_v1_20250707_182610.pth")
agent.q.load_state_dict(checkpoint["parameters"])
agent.optimizer.load_state_dict(checkpoint["optimizer"])

# 4. Simulation (eine Episode)
state, _ = env.reset()
done = False
while not done:
    action = agent.predict_action(state, evaluate=True)  # keine Exploration
    state, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break

env.close()
