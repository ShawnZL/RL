import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Line
from pyecharts import options as opts
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def forward(self, x):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class GreedyAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)

    def select_action(self):
        return np.argmax(self.q_values)


class UCBGreedyAgent:
    def __init__(self, num_actions, c=2.0):
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)
        self.c = c

    def select_action(self):
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(np.sum(self.action_counts) + 1) / (self.action_counts + 1e-8))
        return np.argmax(ucb_values)


class NoisyNetGreedyAgent:
    def __init__(self, num_actions, input_size):
        self.num_actions = num_actions
        self.noisy_net = NoisyLinear(input_size, num_actions)
        self.optimizer = optim.SGD(self.noisy_net.parameters(), lr=0.01)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        q_values = self.noisy_net(state)
        return torch.argmax(q_values).item()


# Environment
num_actions = 5
input_size = 10
env = np.random.rand(input_size)

# Agents
greedy_agent = GreedyAgent(num_actions)
ucb_greedy_agent = UCBGreedyAgent(num_actions)
noisynet_greedy_agent = NoisyNetGreedyAgent(num_actions, input_size)

# Training
num_steps = 1000
rewards_greedy = []
rewards_ucb = []
rewards_noisynet = []

for step in range(num_steps):
    # Random state
    state = np.random.rand(input_size)

    # Greedy Agent
    action_greedy = greedy_agent.select_action()
    reward_greedy = env[action_greedy]
    greedy_agent.q_values[action_greedy] += (reward_greedy - greedy_agent.q_values[action_greedy]) / (step + 1)
    rewards_greedy.append(reward_greedy)

    # UCB Greedy Agent
    action_ucb = ucb_greedy_agent.select_action()
    reward_ucb = env[action_ucb]
    ucb_greedy_agent.q_values[action_ucb] += (reward_ucb - ucb_greedy_agent.q_values[action_ucb]) / (
                ucb_greedy_agent.action_counts[action_ucb] + 1)
    ucb_greedy_agent.action_counts[action_ucb] += 1
    rewards_ucb.append(reward_ucb)

    # NoisyNet Greedy Agent
    action_noisynet = noisynet_greedy_agent.select_action(state)
    reward_noisynet = env[action_noisynet]
    rewards_noisynet.append(reward_noisynet)

    # Update NoisyNet
    state_tensor = torch.FloatTensor(state)
    target_q = torch.FloatTensor([reward_noisynet])
    predicted_q = noisynet_greedy_agent.noisy_net(state_tensor)[action_noisynet]
    loss = F.mse_loss(predicted_q, target_q)
    noisynet_greedy_agent.optimizer.zero_grad()
    loss.backward()
    noisynet_greedy_agent.optimizer.step()

# 计算统计信息
variance_greedy = np.var(rewards_greedy)
mean_greedy = np.mean(rewards_greedy)

variance_ucb = np.var(rewards_ucb)
mean_ucb = np.mean(rewards_ucb)

variance_noisynet = np.var(rewards_noisynet)
mean_noisynet = np.mean(rewards_noisynet)

# 打印统计信息
print("Greedy Agent - Mean: {:.2f}, Variance: {:.2f}".format(mean_greedy, variance_greedy))
print("UCB Greedy Agent - Mean: {:.2f}, Variance: {:.2f}".format(mean_ucb, variance_ucb))
print("NoisyNet Greedy Agent - Mean: {:.2f}, Variance: {:.2f}".format(mean_noisynet, variance_noisynet))

'''
from pyecharts.charts import Line
from pyecharts import options as opts

# 创建 Line 图表
line = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
line.add_xaxis(range(num_steps))  # X 轴为步数

# 添加数据
line.add_yaxis("Greedy", rewards_greedy, label_opts=opts.LabelOpts(is_show=False))  # 取消数据标记的数字
line.add_yaxis("UCB Greedy", rewards_ucb, label_opts=opts.LabelOpts(is_show=False))  # 取消数据标记的数字
line.add_yaxis("NoisyNet", rewards_noisynet, label_opts=opts.LabelOpts(is_show=False))  # 取消数据标记的数字

# 设置标题和标签
line.set_global_opts(
    title_opts=opts.TitleOpts(title="Agent Comparison"),
    xaxis_opts=opts.AxisOpts(type_="value", name="Steps"),
    yaxis_opts=opts.AxisOpts(type_="value", name="Reward"),
)

# 渲染图表
line.render("agent_comparison.html")
'''