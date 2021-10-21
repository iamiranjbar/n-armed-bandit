import numpy as np
from amalearn.agent import AgentBase

class NetGradientAgent(AgentBase):
    def __init__(self, id, environment):
        super(NetGradientAgent, self).__init__(id, environment)
        self.learning_rate = 0.1
        self.arms_count = self.environment.available_actions()
        self.action_counts = [0 for _ in range(self.arms_count)]
        self.prefrences = [0 for _ in range(self.arms_count)]
        self.prev_best_action = 0
        self.constant_best_count = 0
        self.constant_decision_bound = 1000
        self.reward_mean = 0

    def calculate_policy_from_prefrences(self):
        return [np.exp(prefrence) / sum(np.exp(self.prefrences)) for prefrence in self.prefrences]

    def choose_best_action(self):
        policy = self.calculate_policy_from_prefrences()
        return np.random.choice(np.arange(self.arms_count), p=policy)
    
    def update_action_prefrences(self, action, reward):
        policy = self.calculate_policy_from_prefrences()
        self.action_counts[action] += 1
        trial_count = sum(self.action_counts)
        for index in range(len(self.prefrences)):
            if index == action:
                self.prefrences[index] += (self.learning_rate * (reward - self.reward_mean) * (1 - policy[index]))
            else:
                self.prefrences[index] -= (self.learning_rate * (reward - self.reward_mean) * policy[index])
        self.reward_mean = (self.reward_mean * (trial_count - 1) + reward) / trial_count 

    def take_action(self) -> (object, float, bool, object):
        action = self.choose_best_action()
        if action == self.prev_best_action:
            self.constant_best_count += 1
        else:
            self.constant_best_count = 1
            self.prev_best_action = action
        _, reward, _, _ = self.environment.step(action)
        self.update_action_prefrences(action, reward)
        return action, reward

    def get_best_path(self):
        print("Gradient:")
        return np.argmax(self.prefrences)

    def is_finished(self):
        return self.constant_best_count >= self.constant_decision_bound
