import numpy as np
import gym
from gym.spaces import Box

class SafetyHistoryWrapper(gym.Wrapper):

    def __init__(self, env):

        super().__init__(env)
        low = np.concatenate([env.observation_space.low, [0]])
        high = np.concatenate([env.observation_space.high, [1]])
        self.observation_space = Box(low=low, high=high, shape=(env.observation_space.shape[0] + 1,), dtype=np.float32)

    def reset(self):
        observation, info = self.env.reset()
        self.constraint_violated_in_history = 0
        observation = np.concatenate([observation, [self.constraint_violated_in_history]])
        return observation, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate([observation, [self.constraint_violated_in_history]])
        self.constraint_violated_in_history |= info['constraint_violation']
        info['constraint_violated_in_history'] = self.constraint_violated_in_history
        return observation, reward, done, info