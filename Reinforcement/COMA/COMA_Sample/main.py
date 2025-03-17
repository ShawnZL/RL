import torch
import torch.nn as nn
import torch.functional as F

class RNN(nn.Module):
    # 所有agent 共享一个网络 因此 input_shape = obs_shape + n_actions + n_agents（one_hot_code）
    def __init__(self, input_shape, args):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hiddens_dim)
        h = self.rnn(x, h_in)  # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)  # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h

class ComaCritic(nn.Module):
    def __init__(self, input_shape, arglist):
        """
        输入：当前的状态、当前agent的obs、其他agent执行的动作、当前agent的编号对应的one-hot向量、所有agent上一个timestep执行的动作
        输出：当前agent的所有可执行动作对应的联合Q值，一个n_actions维向量
        """
        super(ComaCritic, self).__init__()
        self.arglist = arglist
        self.fc1 = nn.Linear(input_shape, arglist.critic_dim)
        self.fc2 = nn.Linear(arglist.critic_dim, arglist.critic_dim)
        self.fc3 = nn.Linear(arglist.critic_dim, self.arglist.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


def learn(self, batch, max_episode_len, train_step, epsilon):
    ...

    # 计算每个 agent 的 Ｑ 值更新 Critic
    q_values = self._train_critic(batch, max_episode_len, train_step)  # 训练critic网络，并且得到每个agent的所有动作的Ｑ值
    action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # 走一遍softmax得到每个agent的所有动作的概率

    q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的Ｑ值
    pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的概率
    log_pi_taken = torch.log(pi_taken)

    # 计算 baseline 和 R(s, u) 更新 actor
    baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
    advantage = (q_taken - baseline).detach()
    loss = - (advantage * log_pi_taken).sum()

    self.rnn_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.arglist.grad_norm_clip)
    self.rnn_optimizer.step()

    ...
