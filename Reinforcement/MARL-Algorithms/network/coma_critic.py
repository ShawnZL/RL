import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    def forward(self, input):
        self.weight_epsilon.data.normal_()
        self.bias_epsilon.data.normal_()
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(input, weight, bias)


class ComaCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(ComaCritic, self).__init__()
        self.args = args
        # self.fc1 = nn.Linear(input_shape, args.critic_dim)
        # self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        # self.fc3 = nn.Linear(args.critic_dim, self.args.n_actions)
        self.fc1 = NoisyLinear(input_shape, args.critic_dim)
        self.fc2 = NoisyLinear(args.critic_dim, args.critic_dim)
        self.fc3 = NoisyLinear(args.critic_dim, self.args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q