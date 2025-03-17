import numpy as np
import matplotlib.pyplot as plt


class EpsilonGreedyAgent:
    def __init__(self, num_actions, epsilon):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_values)


class UCBAgent:
    def __init__(self, num_actions, confidence_param):
        self.num_actions = num_actions
        self.confidence_param = confidence_param
        self.q_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)

    def select_action(self):
        ucb_values = self.q_values + self.confidence_param * np.sqrt(
            np.log(np.sum(self.action_counts) + 1) / (self.action_counts + 1e-6))
        return np.argmax(ucb_values)


# NoisyNet agent implementation goes here...

# Environment
num_actions = 5
input_size = 10
env = np.random.rand(input_size)

# Agents
epsilon_greedy_agent = EpsilonGreedyAgent(num_actions, epsilon=0.1)
ucb_agent = UCBAgent(num_actions, confidence_param=1.0)
# NoisyNet agent instantiation goes here...

# Training
num_steps = 1000
rewards_epsilon_greedy = []
rewards_ucb = []
# Rewards for NoisyNet agent go here...

for step in range(num_steps):
    # Random state
    state = np.random.rand(input_size)

    # Epsilon-Greedy Agent
    action_epsilon_greedy = epsilon_greedy_agent.select_action()
    reward_epsilon_greedy = env[action_epsilon_greedy]
    epsilon_greedy_agent.q_values[action_epsilon_greedy] += (reward_epsilon_greedy - epsilon_greedy_agent.q_values[
        action_epsilon_greedy]) / (epsilon_greedy_agent.action_counts[action_epsilon_greedy] + 1)
    epsilon_greedy_agent.action_counts[action_epsilon_greedy] += 1
    rewards_epsilon_greedy.append(reward_epsilon_greedy)

    # UCB Agent
    action_ucb = ucb_agent.select_action()
    reward_ucb = env[action_ucb]
    ucb_agent.q_values[action_ucb] += (reward_ucb - ucb_agent.q_values[action_ucb]) / (
                ucb_agent.action_counts[action_ucb] + 1)
    ucb_agent.action_counts[action_ucb] += 1
    rewards_ucb.append(reward_ucb)


# Plotting
plt.plot(rewards_epsilon_greedy, label='Epsilon-Greedy')
plt.plot(rewards_ucb, label='UCB')
# Plot rewards for NoisyNet agent...
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.show()
