import numpy as np
from matplotlib import pyplot as plt

from amalearn.environment import MutliArmedBanditEnvironment
from amalearn.reward import NetReward
from amalearn.agent import NetEGreedyAgent
from amalearn.agent import NetGradientAgent

ARMS_COUNT = 48
EPOCHS_COUNT = 100000

def create_paths():
    paths = []
    layer1_nodes = [1, 2, 3, 4]
    layer2_nodes = [5, 6, 7]
    layer3_nodes = [8, 9, 10, 11]
    for node1 in layer1_nodes:
        for node2 in layer2_nodes:
            for node3 in layer3_nodes:
                new_path = [0, node1, node2, node3, 12]
                paths.append(new_path)
    return paths

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
    for index in range(EPOCHS_COUNT):
        chosen_action, reward = agent.take_action()
        results.append([chosen_action, reward])
        if agent.is_finished():
            print("Final trial: {}".format(index))
            break
    optimal_action = agent.get_best_path()
    print("Best Path Index: {}".format(optimal_action))
    print("Regret: {}".format(agent.calculate_regret(results)))
    optimal_percentages = calculate_optimal_action_percentages(results, optimal_action)
    environment.reset()
    return optimal_percentages, optimal_action

def gradient_run(agent, environment):
    results = []
    for index in range(EPOCHS_COUNT):
        chosen_action, reward = agent.take_action()
        results.append([chosen_action, reward])
        if agent.is_finished():
            print("Final trial: {}".format(index))
            break
    optimal_action = agent.get_best_path()
    print("Best Path Index: {}".format(optimal_action))
    optimal_percentages = calculate_optimal_action_percentages(results, optimal_action)
    environment.reset()
    return optimal_percentages, optimal_action

def run():
    paths = create_paths()
    rewards = [NetReward(path) for path in paths]
    environment = MutliArmedBanditEnvironment(rewards, EPOCHS_COUNT, '1')
    e_greedy_agent = NetEGreedyAgent('1', environment, 0.5, 0.001)
    gradient_agent = NetGradientAgent('2', environment)

    e_greedy_percentages, optimal_action = e_greedy_run(e_greedy_agent, environment)
    print("Best e-greedy Path: {}".format(rewards[optimal_action].path_string()))

    gradient_percentages, optimal_action = gradient_run(gradient_agent, environment)
    print("Best gradient Path: {}".format(rewards[optimal_action].path_string()))
    
    # x_axis = [i for i in range(0, EPOCHS_COUNT, 100)]
    # plt.plot(x_axis, e_greedy_percentages, label="e-greedy")
    # plt.plot(x_axis, gradient_percentages, label="Gradient")
    # plt.xlabel("Trial #")
    # plt.ylabel("Best Action Selection Rate")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    run()
