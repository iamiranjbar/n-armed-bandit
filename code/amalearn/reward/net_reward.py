from amalearn.reward import RewardBase
import numpy as np

BLUE = 0
GREEN = 1
ORANGE = 2

class NetReward(RewardBase):
    def __init__(self, path):
        super(NetReward, self).__init__()
        self.path = path
        self.node_failure_probs = {
            1: 0.1, 2: 0.06, 3: 0.15, 4: 0.50, 5: 0.1, 6: 0.15, 7: 0.65, 8: 0.12, 9: 0.20, 
            10: 0.05, 11: 0.45
        }
        self.edge_colors = {
            (0, 1): GREEN, (0, 2): BLUE, (0, 3): GREEN, (0, 4): ORANGE, (1, 5): GREEN, 
            (1, 6): BLUE, (1, 7): BLUE, (2, 5): BLUE, (2, 6): ORANGE, (2, 7): ORANGE,
            (3, 5): GREEN, (3, 6): ORANGE, (3, 7): BLUE, (4, 5): ORANGE, (4, 6): GREEN, 
            (4, 7): GREEN, (5, 8): ORANGE, (5, 9): ORANGE, (5, 10): BLUE, (5, 11): BLUE,
            (6, 8): GREEN, (6, 9): GREEN, (6, 10): BLUE, (6, 11): BLUE, (7, 8): ORANGE, 
            (7, 9): ORANGE, (7, 10): GREEN, (7, 11): GREEN, (8, 12): BLUE, (9, 12): ORANGE,
            (10, 12): GREEN, (11, 12): ORANGE
        }
        self.color_mappings = {
            BLUE: (0, 2.4),
            GREEN: (4, 4),
            ORANGE: (3, 9.5)
        }
    
    def get_reward(self):
        delay = 0
        for node in self.path:
            if node in self.node_failure_probs:
                delay += (10 * np.random.binomial(1, self.node_failure_probs[node]))
        index = 0
        while index != len(self.path)-1:
            edge = (self.path[index], self.path[index+1])
            edge_delay_mean, edge_delay_variance = self.color_mappings[self.edge_colors[edge]]
            edge_delay_std = np.sqrt(edge_delay_variance)
            delay += np.random.normal(loc=edge_delay_mean, scale=edge_delay_std)
            index += 1
        return -delay

    def path_string(self):
        return "0 -> {} -> {} -> {} -> 12".format(self.path[1], self.path[2], self.path[3])
