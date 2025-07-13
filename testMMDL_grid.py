import gymnasium as gym
import minigrid
import torch
from minigrid.wrappers import FlatObsWrapper
from dqn import DQNAgent, set_seed, evaluate_policy
import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path="configs/agent/", config_name="config", version_base="1.1")

def main(cfg: DictConfig):
    # 1. Umgebung mit Render-Modus erstellen
    env = gym.make(
        "MiniGrid-Empty-8x8-v0", render_mode="human"
    )  # Passe den Namen ggf. an dein Modell an
    env = FlatObsWrapper(env)  # Flatten the observation space

    set_seed(env, 10)

    # 2. Agent initialisieren (Parameter wie beim Training!

    agent = DQNAgent(
            env,
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            lr=cfg.agent.learning_rate,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            target_update_freq=cfg.agent.target_update_freq,
            seed=cfg.seed,
            useNoisyNet=cfg.agent.use_noisy_net,

    )

    # 3. Modell laden (Pfad ggf. anpassen)
    model_dir = os.path.join(
            hydra.utils.get_original_cwd(), "models"
        )
    checkpoint = torch.load(
        os.path.join(model_dir,"dqn_trained_model_MiniGrid_Empty_8x8_v0_20250713_130635.pth"), map_location=torch.device("cpu")
    )
    agent.q.load_state_dict(checkpoint["parameters"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])

    # 4. Simulation (eine Episode)
    agent.q.eval()
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.predict_action(state, evaluate=True)  # keine Exploration
        state, reward, done, truncated, _ = env.step(action)
        print(action,state, reward)
        if done or truncated:
            break

    # agent.q.eval() # Take deterministic actions at test time (important for NoisyNet layers)
    # total_scores = 0
    # for j in range(1):
    #     s, info = env.reset()
    #     done = False
    #     while not done:
    #         a = agent.predict_action(s, evaluate=True)
    #         s_next, r, dw, tr,_ = env.step(a)
    #         done = (dw or tr)

    #         total_scores += r
    #         s = s_next    
    env.close()


if __name__ == "__main__":
    main()