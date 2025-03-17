import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.9
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=ACTIONS
    )
    return table

def choose_action(state, q_table):
    # choose action Sarse always choose the max one
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

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
