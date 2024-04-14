from tabular_rl.envs.random_known_dynamics_env import RandomKnownDynamicsEnv
import numpy as np
import matplotlib.pyplot as plt
from tabular_rl import finite_mdp_utils as fmdp
import gymnasium as gym
from random import choices

class MonteCarlo(RandomKnownDynamicsEnv):
    def __init__(self, S: int, A: int, epsilon: float):
        self.policy = fmdp.get_uniform_policy_for_fully_connected(S, A)
        self.epsilon = epsilon
        RandomKnownDynamicsEnv.__init__(self, S, A)

    def step_by_policy(self, action):
        # Choose the next state using e-greedy exploration
        if np.random.rand() < self.epsilon:  # Explore with probability epsilon
            next_state = np.random.choice(self.S)
        else:  # Exploit with probability 1 - epsilon
            next_state = np.argmax(self.policy[self.current_observation_or_state])
    
        # Get the reward from the rewards table
        reward = self.rewardsTable[self.current_observation_or_state][action][next_state]
    
        # Update the current state
        self.current_observation_or_state = next_state
    
        # Check if the game is over
        gameOver = False
    
        # Return the next state, reward, game over status, and an empty info dictionary
        return next_state, reward, gameOver, {}
        
    def generate_random_trajectory(self, env: gym.Env, num_steps: int, initial_state: int, initial_action: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        list_of_actions = np.arange(env.A)
        taken_actions = np.zeros(num_steps, dtype=int)
        rewards_tp1 = np.zeros(num_steps)
        states = np.zeros(num_steps, dtype=int)

        # Set the initial state and take the initial action
        env.current_observation_or_state = initial_state
        ob, reward, gameOver, history = env.step_by_policy(initial_action)
        rewards_tp1[0] = reward
        taken_actions[0] = initial_action
        states[0] = initial_state

        for t in range(1, num_steps):  # Start from 1 because we've already taken the first step
            states[t] = env.current_observation_or_state
            weights = self.policy[states[t]]
            if np.sum(weights) == 0:
                taken_actions[t] = np.random.choice(list_of_actions)
            else:
                taken_actions[t] = choices(list_of_actions, weights=weights)[0]
            ob, reward, gameOver, history = env.step_by_policy(taken_actions[t])
            rewards_tp1[t] = reward

        return taken_actions, rewards_tp1, states
    
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def mces(self, env: gym.Env, num_steps: int, num_episodes: int) -> np.ndarray:
            """
            Monte Carlo Exploring Starts (MCES) algorithm for policy evaluation and improvement.

            Args:
                env (gym.Env): The environment.
                num_steps (int): The number of steps to take in each episode.
                num_episodes (int): The number of episodes to run.

            Returns:
                tuple: A tuple containing the policy, action-value function, and rewards per episode.

            """
            Q = np.zeros((env.S, env.A))  # Initialize action-value function
            returns_sum = np.zeros((env.S, env.A))  # Sum of returns for each state-action pair
            returns_count = np.zeros((env.S, env.A))  # Count of returns for each state-action pair
            rewards_per_episode = []  # Track total reward per episode

            for i_episode in range(num_episodes):
                if i_episode % 1000 == 0:
                    print("Episode: ", i_episode)
                # Generate an episode with a random starting state-action pair
                initial_state = np.random.randint(env.S)
                initial_action = np.random.randint(env.A)
                taken_actions, rewards_tp1, states = self.generate_random_trajectory(
                    env, num_steps, initial_state, initial_action)

                # Calculate returns and update Q
                G = 0
                gamma = 0.9
                for t in reversed(range(num_steps)):
                    G = rewards_tp1[t] + gamma * G
                    if not (states[t], taken_actions[t]) in zip(states[:t], taken_actions[:t]):  # First-visit MC
                        returns_sum[states[t]][taken_actions[t]] += G
                        returns_count[states[t]][taken_actions[t]] += 1
                        Q[states[t]][taken_actions[t]] = returns_sum[states[t]][taken_actions[t]] / returns_count[states[t]][taken_actions[t]]

                # Update policy
                for s in range(env.S):
                    self.policy[s] = self.softmax(Q[s])

                # Track total reward for this episode
                rewards_per_episode.append(np.sum(rewards_tp1))

            return self.policy, Q, rewards_per_episode
    
    def plot_learning_curve(self, rewards_per_episode, window_size=1):
        # Calcular a média das recompensas por episódio
        rewards_mean = [np.mean(rewards_per_episode[max(0, i-window_size):i+1]) for i in range(len(rewards_per_episode))]

        # Plotar a curva de aprendizado suavizada
        plt.plot(rewards_mean)
        plt.xlabel('Episódio')
        plt.ylabel('Recompensa Média (Média de {} episódios)'.format(window_size))
        plt.show()


if __name__ == "__main__":
    env = MonteCarlo(3, 2, 0.05)
    num_steps = 100
    rewardsTable = np.array([[[1, 1, 1],
                              [1, 1, 80]],
                             [[1, 1, 1],
                              [1, 10, 1]],
                             [[1, 1, 1],
                              [30, 1, 1]]])
    env.rewardsTable = rewardsTable
    #env.nextStateProbability = nextStateProbability
    print("##########Teste##############")
    updated_policy, Q, rewards_per_episode = env.mces(env, num_steps, num_episodes=100)
    print("Updated policy:")
    print(updated_policy)
    print("Q_table:")
    print(Q)
    #print(nextStateProbability)
    #print("Next state probability:")
    #print(env.nextStateProbability)
    env.plot_learning_curve(rewards_per_episode)