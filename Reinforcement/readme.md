# Reinforcement

| Model Free                            | Model based                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| 无需理解环境（Q_Learning，Sarsa，PG） | 先建立模型，再去解释                                         |
| 都是从环境中得到反馈然后从中学习      | 他能通过想象来预判断接下来将要发生的所有情况. 然后选择这些想象情况中最好的那种. 并依据这种情况来采取下一步的策略, |



| 基于概率（Policy-Based） | 基于价值（value-Based）          |
| ------------------------ | -------------------------------- |
| PG                       | Q_Learning，Sarsa                |
| 基于概率做出决策动作     | 根据所做的动作函数，生成价值函数 |

比如在基于概率这边, 有 policy gradients, 在基于价值这边有 q learning, sarsa 等. 而且我们还能结合这两类方法的优势之处, 创造更牛逼的一种方法, 叫做 actor-critic, actor 会基于概率做出动作, 而 critic 会对做出的动作给出动作的价值, 这样就在原有的 policy gradients 上加速了学习过程.

| On-policy                                                    | Off-policy                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| on-policy的算法只能使用当前正在优化的policy生成的数据来进行训练，当你使用一条(state, action, reward, new_state)的数据对policy网络的参数进行了更新之后，这个“正在优化”的policy立即就变了，于是，你就要用它来生成新的一条数据，再继续进行后面的训练，并且你刚才用于训练的那条数据已经“过时”了，不能再使用，需要丢弃掉。 | Off-policy的算法当你使用n条(state, action, reward, new_state)的数据对policy网络的参数进行了更新之后，这个“正在优化”的policy就变了。有很强的延后性 |



# 20230320 Q-Learning

完成Q-Learning算法的完成解析过程，并编写程序完成Q-Learning的应用。同时似然函数这个概念

[文章链接](https://github.com/ShawnZL/Reinforcement/blob/master/Q_Learning/Q_Learning.md)

# 20230321 Sarsa & Sarsa lambda

理解Sarsa的概念，完成对于编写完成Sarsa的代码，**体会Q_learning中Q值跟新和Sarsa的不同**

[文章链接](https://blog.csdn.net/zuzhiang/article/details/103180841)
