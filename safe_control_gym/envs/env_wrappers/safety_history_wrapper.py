import gym

class SafetyHistoryWrapper(gym.Wrapper):

    def reset(self):
        observation = self.env.reset()
        self.constraint_violated_in_history = 0
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.constraint_violated_in_history *= info['constraint_violation']
        info['constraint_violated_in_history'] = self.constraint_violated_in_history
        return observation, reward, done, info