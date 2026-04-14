"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
RANDOM_SEED = 42


class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self):
        """ """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization:
    """ Create plots for comparing the performance of bandit algorithms. """
    def __init__(self, epsilon_greedy_model, thompson_sampling_model):
        self.epsilon_greedy_model = epsilon_greedy_model
        self.thompson_sampling_model = thompson_sampling_model
    
    def plot1(self):
        """ Plot the learning process of both algorithms on linear and logarithmic scales. """
        eg_avg_reward = np.cumsum(self.epsilon_greedy_model.rewards) / (
            np.arange(1, len(self.epsilon_greedy_model.rewards) + 1)
        )
        ts_avg_reward = np.cumsum(self.thompson_sampling_model.rewards) / (
            np.arange(1, len(self.thompson_sampling_model.rewards) + 1)
        )

        optimal_reward = max(
            self.epsilon_greedy_model.optimal_reward,
            self.thompson_sampling_model.optimal_reward
        )

        plt.figure(figsize=(10, 5))
        plt.plot(eg_avg_reward, label="Epsilon-Greedy")
        plt.plot(ts_avg_reward, label="Thompson Sampling")
        plt.axhline(y=optimal_reward, linestyle="--", label="Optimal Reward")
        plt.title("Learning Process of Bandit Algorithms")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(eg_avg_reward, label="Epsilon-Greedy")
        plt.plot(ts_avg_reward, label="Thompson Sampling")
        plt.axhline(y=optimal_reward, linestyle="--", label="Optimal Reward")
        plt.xscale("log")
        plt.title("Learning Process of Bandit Algorithms (Log Scale)")
        plt.xlabel("Number of Trials (log scale)")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()

    def plot2(self):
        """ Plot the cumulative rewards and cumulative regrets of both algorithms. """
        eg_cumulative_rewards = np.cumsum(self.epsilon_greedy_model.rewards)
        ts_cumulative_rewards = np.cumsum(self.thompson_sampling_model.rewards)

        eg_cumulative_regrets = np.cumsum(self.epsilon_greedy_model.regrets)
        ts_cumulative_regrets = np.cumsum(self.thompson_sampling_model.regrets)

        plt.figure(figsize=(10, 5))
        plt.plot(eg_cumulative_rewards, label="Epsilon-Greedy")
        plt.plot(ts_cumulative_rewards, label="Thompson Sampling")
        plt.title("Cumulative Rewards Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(eg_cumulative_regrets, label="Epsilon-Greedy")
        plt.plot(ts_cumulative_regrets, label="Thompson Sampling")
        plt.title("Cumulative Regrets Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Implement the epsilon-greedy algorithm for multi-armed bandit problems. """
    def __init__(self, p, number_of_trials=NUMBER_OF_TRIALS):
        self.p = np.array(p, dtype=float)
        self.number_of_trials = number_of_trials
        self.k = len(self.p)
        self.reward_estimates = np.zeros(self.k)
        self.action_counts = np.zeros(self.k, dtype=int)
        self.rewards = []
        self.regrets = []
        self.selected_bandits = []
        self.algorithm_name = "EpsilonGreedy"
        self.optimal_reward = np.max(self.p)

    def __repr__(self):
        return f"{self.algorithm_name}(bandits={self.k}, trials={self.number_of_trials})"
    
    def pull(self, bandit_index):
        """
    Sample a reward from the selected bandit.

    Args:
        bandit_index (int): Index of the selected bandit.

    Returns:
        float: Observed reward from the selected bandit.

        """
        return np.random.randn() + self.p[bandit_index]

    def update(self, bandit_index, reward):
        """

Update the estimated reward of the selected bandit.

        Args:
            bandit_index (int): Index of the selected bandit.
            reward (float): Observed reward from the selected bandit.

        Returns: 
                None

        """
        self.action_counts[bandit_index] += 1
        n = self.action_counts[bandit_index]
        estimate = self.reward_estimates[bandit_index]
        self.reward_estimates[bandit_index] = estimate + (reward - estimate) / n
    
    def experiment(self):
        """Run the epsilon-greedy experiment for the specified number of trials. """


        for t in range(1, self.number_of_trials + 1):
            epsilon = 1 / t

            if np.random.random() < epsilon:
                bandit_index = np.random.randint(self.k)
            else:
                bandit_index = np.argmax(self.reward_estimates)

            reward = self.pull(bandit_index)
            self.update(bandit_index, reward)

            regret = self.optimal_reward - self.p[bandit_index]

            self.rewards.append(reward)
            self.regrets.append(regret)
            self.selected_bandits.append(bandit_index + 1)

        logger.info(f"{self.algorithm_name} experiment completed.")

    def report(self):
        """ Store the experiment results in a CSV file and log cumulative reward and cumulative regret. """
        results = pd.DataFrame({
            "Bandit": self.selected_bandits,
            "Reward": self.rewards,
            "Algorithm": self.algorithm_name
        })

        results.to_csv("Exported_Data/epsilon_greedy_rewards.csv", index=False)

        cumulative_reward = np.sum(self.rewards)
        cumulative_regret = np.sum(self.regrets)

        logger.info(f"{self.algorithm_name} cumulative reward: {cumulative_reward:.4f}")
        logger.info(f"{self.algorithm_name} cumulative regret: {cumulative_regret:.4f}")

        return results
#--------------------------------------#

class ThompsonSampling(Bandit):
    """Implement the Thompson Sampling algorithm for multi-armed bandit problems with known precision."""
    def __init__(self, p, number_of_trials=NUMBER_OF_TRIALS, precision=1.0):
        self.p = np.array(p, dtype=float)
        self.number_of_trials = number_of_trials
        self.precision = precision
        self.k = len(self.p)

        self.prior_means = np.zeros(self.k)
        self.prior_precisions = np.ones(self.k)

        self.rewards = []
        self.regrets = []
        self.selected_bandits = []
        self.algorithm_name = "ThompsonSampling"
        self.optimal_reward = np.max(self.p)
    
    def __repr__(self):
        return (
            f"{self.algorithm_name}(bandits={self.k}, "
            f"trials={self.number_of_trials}, precision={self.precision})"
        )
    
    def pull(self, bandit_index):
        """

Sample a reward from the selected bandit.

        Args:
            bandit_index (int): Index of the selected bandit.

        Returns:
            float: Observed reward from the selected bandit.

        """
        return np.random.randn() + self.p[bandit_index]
    
    def update(self, bandit_index, reward):
        """
Update the posterior distribution parameters of the selected bandit.

        Args:
            bandit_index (int): Index of the selected bandit.
            reward (float): Observed reward from the selected bandit.

        Returns:
            None
        """
        prior_mean = self.prior_means[bandit_index]
        prior_precision = self.prior_precisions[bandit_index]

        posterior_precision = prior_precision + self.precision
        posterior_mean = (
            prior_precision * prior_mean + self.precision * reward
        ) / posterior_precision

        self.prior_means[bandit_index] = posterior_mean
        self.prior_precisions[bandit_index] = posterior_precision

    def experiment(self):
        """ Run the Thompson Sampling experiment for the specified number of trials."""


        for _ in range(self.number_of_trials):
            sampled_means = np.random.normal(
                loc=self.prior_means,
                scale=np.sqrt(1 / self.prior_precisions)
            )

            bandit_index = np.argmax(sampled_means)
            reward = self.pull(bandit_index)
            self.update(bandit_index, reward)

            regret = self.optimal_reward - self.p[bandit_index]

            self.rewards.append(reward)
            self.regrets.append(regret)
            self.selected_bandits.append(bandit_index + 1)

        logger.info(f"{self.algorithm_name} experiment completed.")
    
    def report(self):
        """ Store the experiment results in a CSV file and log cumulative reward and cumulative regret."""
        results = pd.DataFrame({
            "Bandit": self.selected_bandits,
            "Reward": self.rewards,
            "Algorithm": self.algorithm_name
        })

        results.to_csv("Exported_Data/thompson_sampling_rewards.csv", index=False)

        cumulative_reward = np.sum(self.rewards)
        cumulative_regret = np.sum(self.regrets)

        logger.info(f"{self.algorithm_name} cumulative reward: {cumulative_reward:.4f}")
        logger.info(f"{self.algorithm_name} cumulative regret: {cumulative_regret:.4f}")

        return results

def comparison():
    """Run both bandit algorithms, compare their results, and visualize their performance."""
    np.random.seed(RANDOM_SEED)
    epsilon_greedy = EpsilonGreedy(BANDIT_REWARD)
    thompson_sampling = ThompsonSampling(BANDIT_REWARD)

    epsilon_greedy.experiment()
    thompson_sampling.experiment()

    epsilon_greedy_results = epsilon_greedy.report()
    thompson_sampling_results = thompson_sampling.report()

    combined_results = pd.concat(
        [epsilon_greedy_results, thompson_sampling_results],
        ignore_index=True
    )
    combined_results.to_csv("Exported_Data/combined_rewards.csv", index=False)

    visualization = Visualization(epsilon_greedy, thompson_sampling)
    visualization.plot1()
    visualization.plot2()

    logger.info("Comparison of Epsilon-Greedy and Thompson Sampling completed.")

    return epsilon_greedy, thompson_sampling, combined_results

if __name__ == '__main__':
    comparison()


