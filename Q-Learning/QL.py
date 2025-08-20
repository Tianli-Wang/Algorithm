import time
import numpy as np
import pandas as pd

# 强化学习环境参数
N_STATES = 6  # 状态数量（网格世界的位置数）
ACTIONS = ["left", "right"]  # 可用动作：向左或向右移动
EPSILON = 0.9  # 贪婪策略参数：90%概率选择最优动作，10%概率随机探索
ALPHA = 0.1  # 学习率：控制新经验覆盖旧经验的程度
GAMMA = 0.9  # 折扣因子：未来奖励的重要性权重（越接近1，越重视未来奖励）
MAX_EPISODES = 15  # 最大训练轮数
FRESH_TIME = 0.2  # 环境刷新时间（控制可视化速度）
TerminalFlag = "terminal"  # 终止状态标记


def build_q_table(n_states, actions):
    """创建Q表：初始化所有状态-动作对的价值为0"""
    return pd.DataFrame(
        np.zeros((n_states, len(actions))),  # 创建n_states行、len(actions)列的零矩阵
        columns=actions  # 列名为动作名称
    )


def choose_action(state, q_table):
    """基于ε-贪婪策略选择动作"""
    state_actions = q_table.loc[state, :]  # 获取当前状态下所有动作的Q值
    # 尝试新动作，避免陷入局部最优
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)  # 探索：随机选择动作
    else:
        action_name = state_actions.idxmax()  # 利用：选择Q值最大的动作
    return action_name


def get_env_feedback(S, A):
    """获取环境反馈：根据当前状态和动作，返回下一状态和奖励"""
    if A == "right":  # 向右移动
        if S == N_STATES - 2:  # 当前在倒数第二个位置（状态4）
            S_ = TerminalFlag  # 到达终止状态
            R = 1  # 获得奖励1（到达目标）
        else:
            S_ = S + 1  # 移动到下一个状态
            R = 0  # 普通移动无奖励
    else:  # 向左移动
        S_ = max(0, S - 1)  # 向左移动一格（不能移出左边界）
        R = 0  # 普通移动无奖励
    return S_, R


def update_env(S, episode, step_counter):
    """更新环境显示：可视化智能体位置"""
    env_list = ["-"] * (N_STATES - 1) + ["T"]  # 环境布局：['-', '-', '-', '-', '-', 'T']
    if S == TerminalFlag:  # 到达终止状态
        interaction = f'Episode {episode + 1}: total_steps = {step_counter}'
        print('\r{}'.format(interaction), end='')
        time.sleep(2)  # 暂停显示结果
    else:
        env_list[S] = '0'  # 智能体当前位置
        interaction = ''.join(env_list)  # 使用join将列表转换为字符串
        print('\r{}'.format(interaction), end='')  # 打印当前布局，并通过\r覆盖上一次的输出，形成动画效果
        time.sleep(FRESH_TIME)  # 控制显示速度


def rl():
    """Q-Learning算法主循环"""
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始化Q表

    for episode in range(MAX_EPISODES):  # 训练轮次循环
        step_counter = 0  # 步数计数器
        S = 0  # 初始状态
        is_terminated = False  # 终止标志

        update_env(S, episode, step_counter)  # 初始化环境显示

        while not is_terminated:  # 单轮训练循环
            A = choose_action(S, q_table)  # 基于Q表选择动作
            S_, R = get_env_feedback(S, A)  # 执行动作并获取环境反馈

            q_predict = q_table.loc[S, A]  # 当前Q值预测

            if S_ != TerminalFlag:  # 非终止状态
                # 贝尔曼方程：目标Q值 = 即时奖励 + 折扣因子 × 下一状态的最大Q值
                q_target = R + GAMMA * q_table.loc[S_, :].max()
            else:  # 终止状态
                q_target = R  # 目标Q值仅为即时奖励
                is_terminated = True  # 设置终止标志

            # Q表更新公式：Q(s,a) += 学习率 × (目标Q值 - 当前Q值)
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)

            S = S_  # 状态转移
            update_env(S, episode, step_counter + 1)  # 更新环境显示
            step_counter += 1  # 步数加1

    return q_table


if __name__ == '__main__':
    q_table = rl()  # 执行Q-Learning训练
    print('\nQ-table:\n', q_table)  # 打印最终Q表