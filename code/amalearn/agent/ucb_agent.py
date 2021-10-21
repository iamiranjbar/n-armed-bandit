import numpy as np
from amalearn.agent import AgentBase

class UCBAgent(AgentBase):
    def __init__(self, id, environment):
        super(UCBAgent, self).__init__(id, environment)
        self.trial_count = 0
        self.arms_count = self.environment.available_actions()
        self.expeted_values = [20 for _ in range(self.arms_count)]
        self.action_counts = [1 for _ in range(self.arms_count)]
        self.alpha = 0.88
        self.betha = 0.88
        self.landa = 2.55
        self.money_save = 5000
        self.money_coeefiecient = 0.001
        self.miss_wait_time = 15
        self.exploration_scale = 12

    def calculate_action_ucb(self, action):
        return self.expeted_values[action] + self.exploration_scale * np.sqrt(np.log(self.trial_count)/self.action_counts[action])

    def choose_best_action(self):
        actions_ucb = [self.calculate_action_ucb(action) for action in range(self.arms_count)]
        return np.argmax(actions_ucb)
        
    def get_bus_utility(self, wait_time, bus_arive_time):
        bus_miss_difference = self.miss_wait_time - bus_arive_time
        money_bonus = self.money_save * self.money_coeefiecient
        wait_miss_diffrence = self.miss_wait_time - wait_time
        if bus_arive_time <= self.miss_wait_time:
            return bus_miss_difference ** self.alpha + money_bonus
        else:
            return -self.landa *((-bus_miss_difference) ** self.betha) + money_bonus - self.landa * ((-wait_miss_diffrence)** self.betha)

    def get_taxi_utility(self, wait_time):
        wait_miss_diffrence = self.miss_wait_time - wait_time
        if wait_time <= self.miss_wait_time:
            return wait_miss_diffrence ** self.alpha
        else:
            return -self.landa * ((-wait_miss_diffrence) ** self.betha)

    def get_utility(self, wait_time, bus_arive_time):
        if bus_arive_time <= wait_time:
            return self.get_bus_utility(wait_time, bus_arive_time)
        else:
            return self.get_taxi_utility(wait_time)
    
    def update_action_expected_value(self, action, utitlity):
        action_count = self.action_counts[action]
        new_expected_value = (action_count / (action_count + 1)) * self.expeted_values[action] + (1 / (action_count+1)) * utitlity
        self.expeted_values[action] = new_expected_value
        self.action_counts[action] += 1

    def take_action(self) -> (object, float, bool, object):
        self.trial_count += 1
        action = self.choose_best_action()
        _, reward, _, _ = self.environment.step(action)
        utitlity = self.get_utility(action, reward)
        self.update_action_expected_value(action, utitlity)
        return action, reward

    def get_best_wait_time(self):
        print("UCB:")
        for index, expected_value in enumerate(self.expeted_values):
            print("Wait time: {} => Expected Utility: {}".format(index, expected_value))
        return np.argmax(self.expeted_values)

    def calculate_regret(self, results):
        optimal_action_reward = max(self.expeted_values)
        regret = 0
        for result in results:
            chosen_action, _ = result
            regret += optimal_action_reward - self.expeted_values[chosen_action]
        return regret