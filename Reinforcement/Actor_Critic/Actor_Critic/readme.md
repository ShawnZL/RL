# code改进

使用方法[原文链接](https://hrl.boyuai.com/chapter/2/actor-critic%E7%AE%97%E6%B3%95)

在这里改进了代码，首先是更新了环境，从v0将环境设置为v1

```python
# old
env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
# new
env_name = 'CartPole-v1'
env = gym.make(env_name)
obs, info = env.reset(seed = 0)
```

第二个是从环境得到返回值

```python
state, info = env.reset() # 我们现在只需要前边一个state
```

因为环境会返回两个返回值给我们，但是我们只需使用state，只需要其中的状态返回值

# 学习

第一点为使用actor跟新函数

```python
log_probs = torch.log(self.actor(states).gather(1, actions))
actor_loss = torch.mean(-log_probs * td_delta.detach())
```

在强化学习中，使用策略梯度方法训练一个 Actor-Critic 网络时，我们希望最大化行为的长期累积回报。为了达到这个目标，我们需要定义一个适当的损失函数来更新 Actor 网络的参数。

在给定状态下，Actor 网络根据当前策略选择一个动作，并根据这个动作的概率计算出对应的对数概率（log_probs）。而 td_delta 则代表了时序差分误差，它衡量了当前状态的估计值与目标值之间的差距。

为了最大化累积回报，我们希望增大对应于高回报的动作的概率，并减小对应于低回报的动作的概率。因此，我们希望最小化 actor_loss，而不是最大化。通过乘以负号，我们将最小化 actor_loss 转化为最大化的优化问题。

在具体计算损失时，我们将 log_probs 与 td_delta 相乘，并取其负均值作为损失值。这样做的目的是使高回报动作的概率趋近于1（log_probs 较大），而低回报动作的概率趋近于0（log_probs 较小），从而增大高回报动作被选中的概率，减小低回报动作被选中的概率。

通过最小化 actor_loss，我们可以更新 Actor 网络的参数，以使其产生更优的策略来最大化累积回报。

需要注意的是，这种形式的损失函数仅适用于基于策略梯度的 Actor-Critic 方法，具体的损失函数形式可能会因算法和网络架构的不同而有所变化。

第二个为critic跟新函数

```python
critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
```

在强化学习中，Critic 网络用于估计状态值或动作值函数，帮助评估动作的好坏或状态的价值。在 Actor-Critic 方法中，我们希望最小化 Critic 网络的估计值与目标值之间的均方误差，以逼近真实的价值函数。

在给定状态下，Critic 网络根据当前的参数估计状态值或动作值。而 td_target 则代表了时序差分目标，它是通过当前状态的奖励与下一个状态的估计值乘以折扣因子得到的。td_target 反映了当前状态的长期累积回报的估计值。

为了训练 Critic 网络，我们计算 Critic 网络的估计值与 td_target 之间的均方误差。通过最小化 critic_loss，我们可以更新 Critic 网络的参数，使其更准确地估计状态值或动作值函数。

需要注意的是，这里使用了 torch 中的均方误差损失函数 F.mse_loss()，它用于计算两个张量之间的均方误差。torch.mean() 用于计算均值，将均方误差转化为一个标量值作为损失值。

通过最小化 critic_loss，我们可以更新 Critic 网络的参数，使其更好地逼近真实的价值函数，从而提高策略的性能。