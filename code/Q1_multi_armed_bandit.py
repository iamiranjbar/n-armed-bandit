import numpy as np

from amalearn.environment import MutliArmedBanditEnvironment
from amalearn.reward import GaussianReward
from amalearn.agent import MultiArmedBanditAgent

ARMS_COUNT = 10
RUNS_COUNT = 20
TRIALS_COUNT = 1000


def calculate_optimal_action_percentage(results, optimal_action):
    actions = {i: 0 for i in range(ARMS_COUNT)}
    for result in results:
        chosen_action, _ = result
        actions[chosen_action] += 1
    optimal_action_percentage = actions[optimal_action] / TRIALS_COUNT
    return optimal_action_percentage


def run():
    rewards = [GaussianReward(np.random.normal(0, 1), 1) for _ in range(ARMS_COUNT)]
    environment = MutliArmedBanditEnvironment(rewards, 1000, '1')
    optimal_action_percentages = []
    regrets = []
    for _ in range(RUNS_COUNT):
        results = []
        agent = MultiArmedBanditAgent('1', environment)
        for _ in range(TRIALS_COUNT):
            chosen_action, reward = agent.take_action()
            results.append([chosen_action, reward])
        optimal_action = agent.get_best_action()
        optimal_action_percentage = calculate_optimal_action_percentage(results, optimal_action)
        optimal_action_percentages.append(optimal_action_percentage)
        regret = agent.calculate_regret(results)
        regrets.append(regret)
        environment.reset()
    print("Regrets:\t\t\tmean: {}\tstd: {}".format(np.mean(regrets), np.std(regrets)))
    print("Optimal action percentage:\tmean: {}\tstd: {}".format(np.mean(optimal_action_percentages),
          np.std(optimal_action_percentages)))

            

if __name__ == "__main__":
    run()