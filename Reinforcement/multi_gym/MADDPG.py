import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import utils
from tqdm import tqdm
import sys
import gym
import ma_gym
# from multiagent.environment import MultiAgentEnv
# import multiagent.scenarios as scenarios


# def make_env(scenario_name):
#     # 从环境文件脚本中创建环境
#     scenario = scenarios.load(scenario_name + ".py").Scenario()
#     world = scenario.make_world()
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
#                         scenario.observation)
#     return env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def onehot_from_logits(logits, eps=0.01):
    # 需要将独热格式转换
    ''' 生成最优动作的独热（one-hot）形式 '''
    # 通过比较logits与它们的最大值来确定最可能的动作
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 使用epsilon-贪婪策略在最可能动作和随机动作之间进行选择
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])

def select_action_from(actions, explore):
    if explore:
        # 使用 argmax 选择每个分布中的动作
        selected_actions = torch.argmax(actions, dim=1).tolist()
    else:
        # 将独热编码数组转换为 PyTorch Tensor
        actions_tensor = torch.tensor(actions)

        # 对每个数组，找到值为1的索引
        selected_actions = [torch.nonzero(action).item() for action in actions_tensor]
    return selected_actions


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        # 这里return就是的一个tensor([[0., 0., 1., 0., 0., 0., 0.]], device='cuda:0', grad_fn=<AddBackward0>)
        action = self.actor(torch.tensor(state).to(device))
        # print(f'action is {action}')
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        # print(f'action after is {action}')
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        # print(f'agents is {env.n_agents}')
        for i in range(env.n_agents):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.env = env
        self.lmbda = 0.97
        self.eps = 0.2

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(self.env.n_agents)
        ]
        # states = torch.tensor([states], dtype=torch.float).to(self.device)
        # print(states)
        actions = [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]
        selected_actions = select_action_from(actions, explore)
        # print(selected_actions)
        return actions, selected_actions
        # return [
        #     agent.take_action(state, explore)
        #     for agent, state in zip(self.agents, states)
        # ]

    def update(self, transition_dict, i_agent):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # states 100 * 150
        # actions 100 * 1
        print(next_states.size())
        print(actions.size())
        # 进行拼接
        concatenated_input = concatenated_input = torch.cat([states, actions], dim=1)
        print( concatenated_input.size())
        # Compute TD target for the current agent
        td_target = rewards + self.gamma * self.agents[i_agent].target_critic(concatenated_input) * (1 - dones)
        td_delta = td_target - self.agents[i_agent].critic(states)
        advantage = utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.agents[i_agent].actor(states).gather(1, actions)).detach()

        log_probs = torch.log(self.agents[i_agent].actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        critic_loss = torch.mean(F.mse_loss(self.agents[i_agent].critic(states), td_target.detach()))

        # Adjust critic_input_dim here as well
        critic_input = torch.cat((*states, *actions), dim=1)
        cur_agent = self.agents[i_agent]
        # Perform critic update
        cur_agent.critic_optimizer.zero_grad()
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, td_target.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # Perform actor update
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(states[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, states)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*states, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

        # Update target networks
        self.update_all_targets()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    env = gym.make('ma_gym:Combat-v0', grid_shape=(15, 15), n_agents=2, n_opponents=2)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

def MADDPG_train():
    num_episodes = 10000
    episode_length = 25  # 每条序列的最大长度
    buffer_size = 100000
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 1024
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 100
    minimal_size = 4000

    team_size = 2
    grid_size = (15, 15)
    env_id = 'ma_gym:Combat-v0'
    env = gym.make('ma_gym:Combat-v0', grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
    replay_buffer = utils.ReplayBuffer(buffer_size)

    state_dims = []
    action_dims = []
    # state_dims = env.observation_space[0].shape[0]
    # action_dims = env.action_space[0].n

    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)
    print(state_dims)
    print(action_dims)
    print(critic_input_dim)
    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau)

    win_list = []  # 记录每一轮的回报（return）
    for i in range(10):
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
                    # a_1 = maddpg.take_action(s[0])
                    # a_2 = maddpg.take_action(s[1])
                    actions_multi, actions = maddpg.take_action(states=s, explore=False)
                    next_s, r, done, info = env.step(actions)
                    transition_dict_1['states'].append(s[0])
                    transition_dict_1['actions'].append(actions_multi)
                    transition_dict_1['next_states'].append(next_s[0])
                    transition_dict_1['rewards'].append(r[0] + 100 if info.get('win', False) else r[0] - 0.1)
                    transition_dict_1['dones'].append(False)
                    transition_dict_2['states'].append(s[1])
                    transition_dict_2['actions'].append(actions_multi)
                    transition_dict_2['next_states'].append(next_s[1])
                    transition_dict_2['rewards'].append(r[1] + 100 if info.get('win', False) else r[0] - 0.1)
                    transition_dict_2['dones'].append(False)
                    s = next_s
                    terminal = all(done)
                win_list.append(1 if info.get('win', False) else 0)
                maddpg.update(transition_dict_1, 0)
                maddpg.update(transition_dict_2, 1)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(win_list[-100:])
                    })
                pbar.update(1)
    return_array = np.array(win_list)
    # 每100条轨迹取一次平均
    win_array = np.mean(return_array.reshape(-1, 100), axis=1)

    episodes_list = np.arange(win_array.shape[0]) * 100
    plt.plot(episodes_list, win_array)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"Combat by MADDPG")
    plt.savefig('example1.png')
    plt.show()