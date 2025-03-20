import numpy as np

def compute(P, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape((-1, 1))
    values = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)

    return values

def occupancy(episodes, s, a, timestep_max, gamma):
    rho = 0
    total_times = np.zeros(timestep_max)
    occur_times = np.zeros(timestep_max)
    for episode in range(episodes):
        for i in range(len(episodes)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1

    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho
