import torch
import torch.nn.functional as F
import numpy as np
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import json
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=1).mean()

def calculate_team_individual_similarity(agents, states):
    probs = [agent.actor(torch.tensor(state, dtype=torch.float).to(agent.device)).unsqueeze(0) for agent, state in zip(agents, states)]
    avg_probs = torch.mean(torch.cat(probs, dim=0), dim=0)
    similarities = [cosine_similarity(prob, avg_probs) for prob in probs]
    return torch.tensor(similarities, dtype=torch.float).to(agents[0].device)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)

class PredictionNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(PredictionNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 prediction_lr, lmbda, eps, gamma, curiosity_factor, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.predictor = PredictionNet(state_dim, hidden_dim).to(device)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=prediction_lr)
        self.curiosity_factor = curiosity_factor

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
        predicted_next_states = self.predictor(states)
        prediction_error = F.mse_loss(predicted_next_states, next_states.detach())
        intrinsic_reward = self.curiosity_factor * prediction_error.unsqueeze(-1)
        td_target = rewards + intrinsic_reward + self.gamma * self.critic(next_states) * (1 - dones)

        # td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps,
                            1 + self.eps) * advantage  # 截断
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

def IPPO_train():
    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    lmbda = 0.97
    eps = 0.8
    curiosity_factor = 0.05  # 设置好奇心因子的大小
    prediction_lr = 1e-3  # 设置预测网络的学习率
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    team_size = 2
    grid_size = (15, 15)
    # 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
    env = gym.make('ma_gym:Combat-v0', grid_shape = grid_size, n_agents = team_size, n_opponents = team_size)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    # 两个智能体共享同一个策略
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, prediction_lr, lmbda, eps,
                gamma, curiosity_factor, device)

    win_list = []
    for i in range(1):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                transition_dict_1 = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                transition_dict_2 = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                s = env.reset()
                terminal = False
                while not terminal:
                    a_1 = agent.take_action(s[0])
                    a_2 = agent.take_action(s[1])
                    next_s, r, done, info = env.step([a_1, a_2])
                    similarity_reward = calculate_team_individual_similarity([agent, agent], s)  # 计算相似性奖励
                    transition_dict_1['states'].append(s[0])
                    transition_dict_1['actions'].append(a_1)
                    transition_dict_1['next_states'].append(next_s[0])
                    # transition_dict_1['rewards'].append(
                    #     r[0] + 100 if info.get('win', False) else r[0] - 0.1)
                    transition_dict_1['rewards'] = [r + similarity_reward[0].item() for r in
                                                    transition_dict_1['rewards']]
                    transition_dict_1['dones'].append(False)
                    transition_dict_2['states'].append(s[1])
                    transition_dict_2['actions'].append(a_2)
                    transition_dict_2['next_states'].append(next_s[1])
                    # transition_dict_2['rewards'].append(
                    #     r[1] + 100 if info.get('win', False) else r[1] - 0.1)
                    transition_dict_2['rewards'] = [r + similarity_reward[1].item() for r in
                                                    transition_dict_2['rewards']]
                    transition_dict_2['dones'].append(False)
                    s = next_s
                    terminal = all(done)
                win_list.append(1 if info.get("win", False) else 0)
                agent.update(transition_dict_1)
                agent.update(transition_dict_2)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(win_list[-100:])
                    })
                pbar.update(1)
    win_array = np.array(win_list)
    # Save to JSON
    data_to_save = {
        'win_array': win_array.tolist()  # Convert numpy array to list
    }

    with open('win_array_data.json', 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    # 每100条轨迹取一次平均
    win_array = np.mean(win_array.reshape(-1, 100), axis=1)

    episodes_list = np.arange(win_array.shape[0]) * 100
    plt.plot(episodes_list, win_array)
    plt.xlabel('Episodes')
    plt.ylabel('Win rate')
    plt.title('IPPO on Combat')
    plt.savefig('example.png')
    plt.show()

IPPO_train()
