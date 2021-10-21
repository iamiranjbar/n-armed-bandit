import numpy as np
from matplotlib import pyplot as plt

from amalearn.environment import MutliArmedBanditEnvironment
from amalearn.reward import PositiveGaussianReward
from amalearn.agent import EGreedyAgent
from amalearn.agent import UCBAgent

ARMS_COUNT = 18
BUS_WAIT_MEAN = 8
BUS_WAIT_STD = 3
EPOCHS_COUNT = 100000

def calculate_optimal_action_percentages(results, optimal_action):
    result = []
    actions = {i: 0 for i in range(ARMS_COUNT)}
    for index in range(len(results)):
        chosen_action, _ = results[index]
        actions[chosen_action] += 1
        if (index+1) % 100 == 0:
            optimal_action_percentage = actions[optimal_action] / index
            result.append(optimal_action_percentage)
    return result

def e_greedy_run(agent, environment):
    results = []
    for _ in range(EPOCHS_COUNT):
        chosen_action, reward = agent.take_action()
        results.append([chosen_action, reward])
    optimal_action = agent.get_best_wait_time()
    print("Best Wait Time: {}".format(optimal_action))
    print("Regret: {}".format(agent.calculate_regret(results)))
    optimal_percentages = calculate_optimal_action_percentages(results, optimal_action)
    environment.reset()
    return optimal_percentages

def ucb_run(agent, environment):
    results = []
    for _ in range(EPOCHS_COUNT):
        chosen_action, reward = agent.take_action()
        results.append([chosen_action, reward])
    optimal_action = agent.get_best_wait_time()
    print("Best Wait Time: {}".format(optimal_action))
    print("Regret: {}".format(agent.calculate_regret(results)))
    optimal_percentages = calculate_optimal_action_percentages(results, optimal_action)
    environment.reset()
    return optimal_percentages

def run():
    rewards = [PositiveGaussianReward(BUS_WAIT_MEAN, BUS_WAIT_STD) for _ in range(ARMS_COUNT)]
    environment = MutliArmedBanditEnvironment(rewards, EPOCHS_COUNT, '1')
    e_greedy_agent = EGreedyAgent('1', environment, 0.001)
    ucb_agent = UCBAgent('2', environment)
    
    e_greedy_percentages = e_greedy_run(e_greedy_agent, environment)
    ucb_percentages = ucb_run(ucb_agent, environment)
    x_axis = [i for i in range(0, EPOCHS_COUNT, 100)]

    plt.plot(x_axis, e_greedy_percentages, label="e-greedy")
    plt.plot(x_axis, ucb_percentages, label="UCB")
    plt.xlabel("Trial #")
    plt.ylabel("Best Action Selection Rate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()
