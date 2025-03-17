import torch
import torch.nn as nn
import torch.optim as optiom

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # 很多初始化策略都是为了保持每层的分布不变, 而BN是通过增加归一化层使得每层数据分布保持在N(0, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        # 归一化层
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        # BN是对batch的维度去做归一化，也就是针对不同样本的同一特征做操作。
        # LN是对hidden的维度去做归一化，也就是针对单个样本的不同特征做操作。因此LN可以不受样本数的限制。
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optiom.Adam(self.parameters(), lr = alpha)

        self.apply(weight_init())
        # self.to(device)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.action(x)) # 激活函数tan(x)

        return action

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim) # 加上效果好，为什么？
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(action_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optiom.Adam(self.parameters(), lr = beta, weight_decay = 0.001)
        self.apply(weight_init)
        # self.to(device)

    def forward(self, state, action):
        x_s = torch.relu(self.ln1(self.fc1(state)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(action)
        x = torch.relu(x_s + x_a)
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
       self.load_state_dict(torch.load(checkpoint_file))

