import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """ 伯努利多笔老虎机，输入k表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(low=0, high=1, size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        # 玩家选择k号拉杆箱后，根据老虎机获取奖励的概率返回1或0
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    """ multi arm bandit"""
    # MAB问题目标是最大化累积奖励，等价于最小化累积懊悔，懊悔是当前最优拉杆k-当前拉杆的动作a
    def __init__(self, bandit):
        self.bandit = bandit
        # 每根拉杆的尝试次数
        self.counts = np.zeros(self.bandit.K)
        # 积累懊悔
        self.regret = 0
        self.actions = []
        # 记录每一步的累积懊悔
        self.regrets = []

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon

        # 初始化拉动所有拉杆的期望奖励估值
        # 结果是一个列表，长度为self.bandit.K，每个元素的值都是init_prob
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            # 选择哪一个去拉杆
            k = np.random.randint(0, self.bandit.K)
        else:
            # 选择期望奖励估值最大的拉杆
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        # 使得增长更加接近于r
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.coef = coef
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.total_count + 1)))
        k = np.argmax(ucb) # 选出上置信界最大的拉杆的索引
        r = self.bandit.step(k)
        self.estimates[k] = 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        sample = np.random.beta(self._a, self._b) # 按照Beta分布采样一组奖励样本
        k = np.argmax(sample)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] -= (1 - r)
        return k

K = 10
bandit_10_arm = BernoulliBandit(K)
np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])
