import sys
import os
import gym
import numpy as np
import torch
import torch.optim as optim
import datetime
from networks import Actor, Critic
import matplotlib.pyplot as plt

from torch.distributions import Categorical
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2C:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.actor = Actor(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                           fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic = Critic(beta=beta, state_dim=state_dim,
                             fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.loss = nn.MSELoss()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

    def choose_action(self, observation):
        # self.actor.eval()
        # state = torch.tensor(observation, dtype=torch.float).to(device)
        a = self.actor.forward(observation)
        dist = Categorical(a)
        action = dist.sample() # 可以采取的action
        log_prob = dist.log_prob(action) # 每种action的概率

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, state, state_, reward):
        # 使用Critic网络估计状态值
        v = self.critic(state)
        v_ = self.critic(state_)

        critic_loss = self.loss(self.gamma * v_ + reward, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td_error = self.gamma * v_ + reward - v # 计算td误差
        loss_actor = -log_prob * td_error.detach() # 这一行计算演员（actor）损失，它与动作概率log_prob和TD误差td_error之间的乘积有关。
        # td_error.detach()用于分离TD误差的计算，以确保梯度不会传播到演员网络。
        self.actor_optim.zero_grad() # 这里清除演员网络的先前步骤中的梯度信息，将所有梯度设置为0.
        loss_actor.backward()
        self.actor_optim.step()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    model = A2C(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n, actor_fc1_dim=400, actor_fc2_dim=300,
                critic_fc1_dim=400, critic_fc2_dim=300, gamma=0.99, tau=0.005, action_noise=0.1,
                policy_noise=0.2, policy_noise_clip=0.5, delay_time=2)

    reward_history = []
    for episode in range(200):
        state, _ = env.reset() # 获取当前环境信息
        print(state)
        env.render() # 界面可视化
        done = False
        ep_r = 0
        while not done:
            # 通过Actor_Critic算法对于当前环境作出行动
            action, log_prob = model.choose_action(state)
            state_, reward, done, truncated, info = env.step(action)

            # 计算当前的reward 总和
            ep_r += reward

            # 训练模型
            model.learn(log_prob, state, state_, reward)

            # 更新环境
            state = state_
        reward_history.append(ep_r)
        print(f"episode:{episode} ep_r:{ep_r}")
    plt.plot(reward_history)
    plt.show()

