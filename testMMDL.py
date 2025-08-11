import gymnasium as gym
import torch
from dqn import DQNAgent, set_seed, evaluate_policy
from wrapper import FlatObsImageOnlyWrapper
from minigrid.wrappers import FlatObsWrapper
from hydra.core.hydra_config import HydraConfig
import hydra
import os
from typing import Any, Dict, List, Tuple
from omegaconf import DictConfig


@hydra.main(config_path="configs/agent/", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    timestamp = "20250808_094716"
    working_dir =os.path.join(hydra.utils.get_original_cwd(),"RAW_Data",timestamp, "models")


    files = os.listdir(working_dir)
    if len(files) == 1:
        model = os.path.join(working_dir,files[0])
    else:
        print(f"Error: Expected 1 file, found {len(files)} files")
        model = None


    # 1) build env
    env = gym.make(cfg.env.name, render_mode="human")
    if "MiniGrid" in cfg.env.name:
            env = FlatObsImageOnlyWrapper(env)    #env = FlatObsWrapper(env)
    set_seed(env, cfg.seed)

    # 2) map config → agent kwargs
    agent_kwargs = dict(
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=10,
        useNoisyNet=cfg.agent.use_noisy_net,
        useNoiseReduction=False,
        minReward = cfg.agent.minReward,
        maxReward = cfg.agent.maxReward
    )
    
    agent = DQNAgent(env, **agent_kwargs)

    # 3. Modell laden
    checkpoint = torch.load(model)
    agent.q.load_state_dict(checkpoint["parameters"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])

    # 4. Simulation (eine Episode)
    score = evaluate_policy(env, agent, turns=10)
    print(score)

    env.close()


if __name__ == "__main__":
    main()