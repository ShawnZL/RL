import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数


# 定义不同的gamma值
gamma_values = [0.95, 0.98, 0.99]

# 共享的超参数
actor_lr = 0.001
critic_lr = 0.001
num_episodes = 1500
hidden_dim = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境和 Agent
env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 创建三个子图
fig, axes = plt.subplots(1, len(gamma_values), figsize=(15, 5), sharey=True)

# 循环尝试不同的gamma值
for i, gamma in enumerate(gamma_values):
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        gamma, device)
    return_list = utils.train_on_policy_agent(env, agent, num_episodes)

    # 归一化处理
    normalized_returns = (np.array(return_list) -
                          np.min(return_list)) / (
                                 np.max(return_list) -
                                 np.min(return_list))

    episodes_list = list(range(len(normalized_returns)))

    # 绘制子图
    axes[i].plot(episodes_list, normalized_returns, label=f'gamma={gamma}')
    axes[i].set_title(f'Gamma = {gamma}')

# 设置共同的横坐标和纵坐标标签
for ax in axes:
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns (0-1)')

# 设置整体标题
plt.suptitle('RL with Different Gamma values')

# 设置子图之间的间距
plt.tight_layout()

plt.show()
