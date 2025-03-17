import gym
import numpy as np
import torch
import torch.nn.functional as F
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # 激活函数
        x = F.softmax(self.fc2(x), dim=1)
        return x

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO():
    # 采取截断的方法
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.lmbda = lmbda
        self.eps = eps
        self.gamma = gamma
        self.epochs = epochs # 一条用于训练的轮数
        self.device = device

    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state) # action_dim 概率分布
        action_dist = torch.distributions.Categorical(probs) # 对传入的prob作出归一化处理
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)

        # 时间差分方程 TD_error = r + γV(s') - V(s)
        td_target = rewards + self.gamma * (1 - dones) * self.critic(next_states)
        # td_delta 每个时间步的优势
        td_delta = td_target - self.critic(states)
        # 使用 λ-return 方法平滑 TD 误差
        advantages = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # gather 选择分布概率最大的
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach() # detach 不计算梯度

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = (log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages # 截断
            # 选择其中一个好动作，因为我们希望 最大化 目标函数，计算所有样本的平均损失
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            # 这里使用 均方误差 (MSE Loss) 来衡量预测值与目标值之间的偏差
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
