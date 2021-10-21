import numpy as np
from amalearn.agent import AgentBase

class NetEGreedyAgent(AgentBase):
    def __init__(self, id, environment, epsilon, learning_rate):
        super(NetEGreedyAgent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.arms_count = self.environment.available_actions()
        self.expeted_values = [0 for _ in range(self.arms_count)]
        self.action_counts = [0 for _ in range(self.arms_count)]
        self.prev_best_action = 0
        self.constant_best_count = 0
        self.constant_decision_bound = 1000

    def choose_best_action(self):
        p = np.random.random()
        if p < self.epsilon:
            return np.random.choice(self.arms_count)
        else:
            return np.argmax(self.expeted_values)
    
    def update_action_expected_value(self, action, reward):
        action_count = self.action_counts[action]
        new_expected_value = (action_count / (action_count + 1)) * self.expeted_values[action] + (1 / (action_count+1)) * reward
        self.expeted_values[action] = new_expected_value
        self.action_counts[action] += 1

    def take_action(self) -> (object, float, bool, object):
        action = self.choose_best_action()
        if action == self.prev_best_action:
            self.constant_best_count += 1
        else:
            self.constant_best_count = 1
            self.prev_best_action = action
        _, reward, _, _ = self.environment.step(action)
        self.update_action_expected_value(action, reward)
        self.epsilon -= self.learning_rate
        return action, reward

    def get_best_path(self):
        print("Epsilon-Greedy:")
        for index, expected_value in enumerate(self.expeted_values):
            print("Path: {} => Expected Delay: {}".format(index, expected_value))
        return np.argmax(self.expeted_values)

    def calculate_regret(self, results):
        optimal_action_reward = max(self.expeted_values)
        regret = 0
        for result in results:
            chosen_action, _ = result
            regret += optimal_action_reward - self.expeted_values[chosen_action]
        return regret

    def is_finished(self):
        return self.constant_best_count >= self.constant_decision_bound