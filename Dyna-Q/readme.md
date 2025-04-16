强化学习分为两种方式：model-based & model-free

其中大部分算法都是model-free，无模型的强化学习根据智能体与环境交互采样到的数据直接进行策略提升或者价值估计。

model-based 比较多的是动态规划

Dyna-Q 算法也是非常基础的基于模型的强化学习算法，不过它的环境模型是通过采样数据估计得到的。



每一次迭代都是用Q-learning

在每次迭代中，便利当前已经存储的环境（学习现在环境交互的数据内容）便利N次进行Q-learning



初始化 \( Q(s, a) \)，初始化模型 \( M(s, a) \)

1. 对于每个序列 \( e = 1 \) 到 \( E \)：
   - 获取初始状态 \( s \)
   - 对于每个时间步 \( t = 1 \) 到 \( T \)：
     - 使用 ε-贪婪策略根据 \( Q \) 选择动作 \( a \)
     - 获取环境反馈 \( r, s' \)
     - 更新 \( Q \) 值：
       Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
     - 更新模型 \( M \)：
       M(s, a) \leftarrow r, s'
     - 对于次数 \( n = 1 \) 到 \( N \)：
       - 随机选择一个曾访问过的状态 \( s_m \)
       - 选择在状态 \( s_m \) 执行过的动作 \( a_m \)
       - 从模型中获取 \( r_m, s'_m \)：
         r_m, s'_m \leftarrow M(s_m, a_m)
       - 更新 \( Q \) 值：
         Q(s_m, a_m) \leftarrow Q(s_m, a_m) + \alpha [r_m + \gamma \max_{a'} Q(s'_m, a') - Q(s_m, a_m)]
     - 状态更新 \( s \leftarrow s' \)



