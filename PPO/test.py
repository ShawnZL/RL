import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.device = device
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 这是一个实例变量，用于存储优化器对象。这个优化器将用于更新 self.actor 网络的参数。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps # clip 参数
        self.gamma = gamma

    def take_action(self, state):
        state = torch.tensor([state], dtype = torch.float32).to(self.device)
        probs = self.actor(state)
        # Categorical 是torch中一个类，表示离散的多项分布
        # 主要方法 sample() 随机采样；log_prob(value) 对于value计算对数概率；probs 是分布的概率向量 logits 是分布的对数几率，通常是在 softmax 转换之前的网络输出
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        # view reshape tensor 第二维度为1
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones']).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        # TD 误差，计算td目标和当前价值的差异
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 在固定的采样上多次调整数据，从而获取更好的性能
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
            # 本来是log，然后更换为exp抵消log
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 -self.eps, 1+self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2)) # ppo 损失函数，限制更新幅度
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))

            # 清除梯度缓存
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 计算损失的梯度
            actor_loss.backward()
            critic_loss.backward()
            # 更新模型参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()

