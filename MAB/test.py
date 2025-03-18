import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, n_arms):
        self.probs = np.random.uniform(n_arms)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = n_arms

    def step(self, k):
        if np.random.random() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(bandit.K)
        self.regret = 0
        self.regrets = []
        self.actions = []

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs(k)
        self.regrets.append(self.regret)

    def run_one_step(self, k):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.update_regret(k)
            self.actions.append(k)
            self.counts[k] += 1

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        """ epsilon贪婪算法,继承Solver类 """
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.choice(self.bandit.K)
        else:
            k = np.argmax(self.estimates) # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k

class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.choice(self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.coef = coef
        self.estimates = np.array([init_prob] * bandit.K)

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))) # 计算上置信界
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K) # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K) # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] -= (1 - r)
        return k

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] -= (1 - r)