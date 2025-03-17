import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3
LAMBDA = 0.5

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

# draw the picture
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

# Sarsa lambda
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    E_table = pd.DataFrame(np.zeros((N_STATES, len(ACTIONS))), columns=ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        A = choose_action(S, q_table)
        while not is_terminated:
            S_,R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                A_ = choose_action(S_, q_table)
                delta = R + GAMMA * q_table.loc[S_, A_] - q_predict
                E_table.iloc[S,:] = 0
                E_table.loc[S, A] = 1
                # 对于矩阵E_table的遍历
                """
                # for 循环
                for i in range(N_STATES):
                    for As in ACTIONS:
                        if (E_table.loc[i, As]) != 0:
                            q_table.loc[i, As] += ALPHA * delta * E_table.loc[i, As]
                            E_table.loc[i, As] = GAMMA * LAMBDA * E_table.loc[i, As]
                """
                # 直接使用矩阵乘法，优化结构
                q_table += ALPHA * delta * E_table
                E_table = GAMMA * LAMBDA * E_table
            else:
                q_target = R
                is_terminated = True
                q_table.loc[S,A] += ALPHA * (q_target - q_predict)

            S = S_
            A = A_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
            # print(E_table)
    return q_table



if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

