import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box

class FlatObsImageOnlyWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_shape = env.observation_space['image'].shape
        obs_size = np.prod(obs_shape)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(obs_size,),
            dtype='uint8'
        )

    def observation(self, obs):
        return obs["image"].flatten()