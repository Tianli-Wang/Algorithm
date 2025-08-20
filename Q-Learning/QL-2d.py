import time
import logging
import gym
import numpy as np
import pandas as pd

GRID_SIZE = 4
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
MAX_EPISODE = 150
EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
TERMINAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)


def build_table(grid_size, actions):
    rows = pd.MultiIndex.from_product(
        [range(grid_size), range(grid_size)],
        names=['x', 'y']
    )
    return pd.DataFrame(
        np.zeros((grid_size ** 2, len(actions))),
        index=rows,
        columns=actions
    )


def choose_action(state, q_table):
    """greedy method for choosing the best action"""
    state_actions = q_table.loc[state, :]

    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_back(state, action):
    x, y = state
    if action == 'UP':
        y_new = y - 1
        x_new = x
    elif action == 'DOWN':
        y_new = y + 1
        x_new = x
    elif action == 'LEFT':
        y_new = y
        x_new = x - 1
    elif action == 'RIGHT':
        y_new = y
        x_new = x + 1

    is_out_of_bounds = (
            x_new < 0 or x_new >= GRID_SIZE or y_new < 0 or y_new >= GRID_SIZE
    )

    #  边界约束
    x_new = max(0, min(GRID_SIZE - 1, x_new))
    y_new = max(0, min(GRID_SIZE - 1, y_new))

    state_new = (x_new, y_new)

    if state_new == TERMINAL_POS:
        r = 100
    elif is_out_of_bounds:
        r = -10
    else:
        r = -0.1

    return state_new, r


def log_q_table(table, title="Q-Table:"):
    """记录DataFrame格式的Q表到日志"""
    logging.info("%s\n%s", title, table)


def rl():
    q_table = build_table(GRID_SIZE, ACTIONS)
    # log_q_table(q_table, "初始Q表:")
    for episode in range(MAX_EPISODE):
        state = (np.random.randint(0, GRID_SIZE - 1), np.random.randint(0, GRID_SIZE - 1))
        terminated = False  # 是否到达终点
        step = 0
        while not terminated:
            action = choose_action(state, q_table)
            next_state, r = get_env_back(state, action)

            if next_state != TERMINAL_POS:
                q_target = r + GAMMA * q_table.loc[next_state, :].max()
            else:
                q_target = r
                terminated = True
            # 测下一步用到的
            q_predict = q_table.loc[state, action]
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            log_q_table(q_table, "Q-Table after episode %d" % episode)
            state = next_state

    return q_table


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            # logging.FileHandler('QL.log'),  # 输出到文件
            logging.StreamHandler()  # 输出到终端
        ]
    )

    q_table_final = rl()
    log_q_table(q_table_final, "final Q_table:")
    # print(build_table(GRID_SIZE, ACTIONS))
    # q_table_print = build_table(GRID_SIZE, ACTIONS)
    # log_q_table(q_table, "初始Q表:")
    # print('\nQ-table:\n', q_table)  # 打印最终Q表
