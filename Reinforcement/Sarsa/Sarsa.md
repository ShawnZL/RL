# Sarsa

Sarsa 算法与 Q-Learning 算法相似，也是利用 Q 表来选择动作，唯一不同的是两者 Q 表的更新策略不同。该算法由于更新一次动作值函数需要用到 5 个量 ( s , a , r , s ′ , a ′ ) ，所以被称为 Sarsa 算法。

跟新Q表的公式如下所示
$$
Q(S_t,A_t)<-Q(S_t,A_t)+α[R_{t+1}+γ*Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]
$$
其中 Q(St+1, At+1)是下一时刻的状态和实际采取的行动对应的 Q 值；而在 Q-Larning 中是下一时刻的状态对应的 Q 值的最大值，但是在实际中可能不采用该最大值对应的动作。Sarsa 算法和 Q-Learning 算法除了在 Q-target 上有所不同，其他的都一样。

Sarsa 算法是on-policy学习算法，它只有一个策略，使用贪婪选择出Q(St,At) 和 Q(St+1′, At+1′) 。但是Q_learning是off-policy算法，Q(St,At) 使用贪婪算法，计算下一步的时候选择最大值。

## Sarsa算法流程



![](https://github.com/ShawnZL/Reinforcement/raw/master/picture/Sarsa_pic1.png)

初始化 Q 表（令其值为 0）

对于每个 episode（回合）：

 1. 初始化状态 s

 2. 在当前状态 s 的所有可能动作中选取一个动作 a （以 ϵ \epsilonϵ 的概率安装 Q 表数值最大的动作行动，以 1 − ϵ 1-\epsilon1−ϵ 的概率随机行动）

 3. 如果当前状态 s 不是终止状态，则重复执行以下步骤：

 （1）执行动作 a 并得到下一个状态 s‘ 和相应的奖励 r

 （2）在当前状态 s’ 的所有可能动作中选取一个动作 a’

 （3）更新 Q 表：Q ( s , a ) ← Q ( s , a ) + α [ r + γ ⋅ Q ( s ′ , a ′ ) − Q ( s , a ) ] 

 （4）更新状态和动作：s=s’， a=a’

```python
# Sarsa
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_couter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_couter)
        A = choose_action(S, q_table) # 首先根据随机动作选择一个动作
        # A 代表下一个行为，一定要先放置在外边
        while not is_terminated:
            S_, R = get_env_feedback(S, A) # 下一个的动作，奖励
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                A_ = choose_action(S_, q_table)
                q_target = R + GAMMA * q_table.loc[S_, A_]
                # 注意Q_learning 选择的方法是选择最大值
                # 但是Sarsa选择的是还是贪婪算法选择行为
                """
                q_target = R + GAMMA * q_table.iloc[S_,:].max()
                """
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)

            S = S_
            A = A_
            update_env(S, episode, step_couter + 1)
            step_couter += 1
    return q_table

```

