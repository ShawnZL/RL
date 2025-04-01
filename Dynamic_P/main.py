import copy

class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.nrow * self.env.ncol)]

        self.theta = theta # 策略评估收敛域值
        self.gamma = gamma

    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.nrow * self.env.ncol):
                qsa_list = [] # 开始计算状态s下所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        qsa += p * (reward + self.gamma * self.v[next_state] * (1 - done))
                        # 奖励和下一个状态有关，所以需要和状态转移概率想乘
                    qsa_list.append(qsa)
                new_v[s] = sum(qsa_list) # 状态价值函数和动作价值函数的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break # 满足收敛条件，退出评估
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_update(self): # 策略提升
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, reward, done = res
                    qsa += p * (reward + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) # 计算有几个是最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self): # 策略迭代
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_update()
            if old_pi == new_pi: break

class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.theta = theta
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.nrow * self.env.ncol)]

    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.nrow * self.env.ncol):
                qsa_list = [] # 开始计算状态s下的所有Q(s,a)
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        qsa += p * (reward + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)

                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
            cnt += 1
        print("价值迭代一共进行了%d" % cnt)
        self.get_policy()

    def get_policy(self): # 根据价值函数导出一个贪婪策略
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, reward, done = res
                    qsa += p * (reward + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
