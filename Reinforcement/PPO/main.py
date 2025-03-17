import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Hyperparameters
lr = 0.001
gamma = 0.99
epsilon = 0.2
epochs = 10
iterations = 200
batch_size = 64

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def ppo(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    model = ActorCritic(num_inputs, num_outputs)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(iterations):
        state = env.reset()
        rewards = []
        log_probs = []
        values = []
        states = []
        actions = []
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs, value = model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            states.append(state)
            actions.append(action)
            state = next_state

        returns = compute_returns(rewards, gamma)
        advantages = [R - value for R, value in zip(returns, values)]

        for _ in range(epochs):
            for idx in range(0, len(rewards), batch_size):
                returns_batch = torch.tensor(returns[idx:idx+batch_size], dtype=torch.float32)
                advantages_batch = torch.tensor(advantages[idx:idx+batch_size], dtype=torch.float32)
                log_probs_batch = torch.stack(log_probs[idx:idx+batch_size])

                new_action_probs, new_values = model(torch.tensor(states[idx:idx+batch_size], dtype=torch.float32))
                new_log_probs = torch.log(new_action_probs[range(batch_size), actions[idx:idx+batch_size]])
                ratios = torch.exp(new_log_probs - log_probs_batch)
                clipped_ratios = torch.clamp(ratios, 1-epsilon, 1+epsilon)

                policy_loss = -torch.min(ratios * advantages_batch, clipped_ratios * advantages_batch).mean()
                value_loss = 0.5 * (returns_batch - new_values.squeeze()).pow(2).mean()
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Iteration {i}: Reward = {sum(rewards)}")

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    ppo(env)
