from amalearn.reward import RewardBase
import numpy as np

class PositiveGaussianReward(RewardBase):
    def __init__(self, mean, std):
        super(PositiveGaussianReward, self).__init__()
        self.mean = mean
        self.std = std

    def get_reward(self):
        reward = np.random.normal(loc=self.mean, scale=self.std)
        while reward < 0:
            reward = np.random.normal(loc=self.mean, scale=self.std)
        return reward
