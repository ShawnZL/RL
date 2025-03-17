# 论文

但是我们将展示现有方法中使用的基于 RL 的目标完全可以通过一个简单的二元交叉熵（Binary Cross-Entropy，BCE）目标来优化，从而大大简化偏好学习的流程。

DPO 更新增加了良好响应与不良响应的相对对数概率，但它结合了一个动态的、每个样本的重要性权重，用于防止在朴素概率比（naive probability ratio）目标下会发生的模型退化。类似现有的方法，DPO 也依赖一个理论上的偏好模型（比如 Bradley-Terry 模型）以衡量给定奖励函数和经验偏好数据间的一致性。

**现有方法使用偏好模型定义偏好损失来训练奖励模型，然后再训练一个用于优化所学奖励模型的策略，而 DPO 则是使用变量的变化把偏好损失直接定义为策略函数。**

现在我们可以将人类偏好数据的概率表示为最优策略而不是奖励模型，这样我们就可以为参数化的策略$\pi_{θ}$ 构建最大似然目标。

公式5-2

通过这种方式，我们同时跳过了显示的奖励建模步骤，并避免了强化学习的过程。此外，由于我们的过程等价于拟合重参数化的 Bradley-Terry 模型，因此它具有一些理论基础，比如在合适的偏好数据分布假设下的一致性。在第 5 节中，我们将进一步讨论 DPO 相关的理论基础与其他工作之间的关系。

[公式推导链接](https://zhuanlan.zhihu.com/p/634705904)

理解RLHF和DPO的差异，本质就是理解强化学习的value estimation和reward的差异。DPO就相当于直接用reward做对策略做正相关优化，而且还是贪心优化。而RL的value estimation则是expected future rewards，相当于动态规划的backup table值而不是贪心值，RL难其实就是没拟合好value estimation，用欠佳的监督对策略做正相关优化。



# code

[Link](https://github.com/eric-mitchell/direct-preference-optimization/tree/main)

