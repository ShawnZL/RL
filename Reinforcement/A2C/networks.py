import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class Actor(nn.Module):
    ''' Actor
    '''
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.a = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        state = T.tensor(state, dtype=T.float).to(device)
        x = T.relu(self.ln1(self.fc1(state)))
        x = T.relu(self.ln2(self.fc2(x)))
        action = T.softmax(self.a(x), dim=-1)
        return action

class Critic(nn.Module):
    ''' 评论家 返回value
    '''
    def __init__(self, beta, state_dim, fc1_dim, fc2_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state):
        state = T.tensor(state, dtype=T.float).to(device)
        x = T.relu(self.ln1(self.fc1(state)))
        x = T.relu(self.ln2(self.fc2(x)))
        q = self.q(x)

        return q
