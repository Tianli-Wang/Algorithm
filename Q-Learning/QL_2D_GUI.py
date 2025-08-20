import time
import logging
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 环境参数
GRID_SIZE = 4
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
MAX_EPISODE = 150
EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
TERMINAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)
CELL_SIZE = 100  # 网格单元格大小


class QLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning网格链路GUI")
        self.root.geometry("1000x600")

        # 初始化Q表
        self.q_table = self.build_table(GRID_SIZE, ACTIONS)

        # 创建界面组件
        self.create_widgets()

        # 状态变量
        self.episode = 0
        self.state = (0, 0)
        self.terminated = False
        self.step = 0
        self.is_training = False
        self.history = []  # 记录每一步的状态和动作

        # 初始化可视化
        self.update_grid_visualization()
        self.update_q_table_display()

    def build_table(self, grid_size, actions):
        """创建Q表"""
        rows = pd.MultiIndex.from_product(
            [range(grid_size), range(grid_size)],
            names=['x', 'y']
        )
        return pd.DataFrame(
            np.zeros((grid_size ** 2, len(actions))),
            index=rows,
            columns=actions
        )

    def choose_action(self, state):
        """基于ε-贪婪策略选择动作"""
        state_actions = self.q_table.loc[state, :]

        if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
            action_name = np.random.choice(ACTIONS)
        else:
            action_name = state_actions.idxmax()
        return action_name

    def get_env_back(self, state, action):
        """获取环境反馈"""
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

        # 边界约束
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

    def create_widgets(self):
        """创建界面组件"""
        # 左侧：网格可视化
        self.grid_frame = tk.Frame(self.root, width=CELL_SIZE * GRID_SIZE, height=CELL_SIZE * GRID_SIZE)
        self.grid_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.grid_canvas = tk.Canvas(
            self.grid_frame,
            width=CELL_SIZE * GRID_SIZE,
            height=CELL_SIZE * GRID_SIZE,
            bg="white"
        )
        self.grid_canvas.pack(fill=tk.BOTH, expand=True)

        # 右侧：控制面板和Q表
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板
        self.control_frame = tk.Frame(self.right_frame)
        self.control_frame.pack(fill=tk.X, pady=10)

        self.start_btn = tk.Button(
            self.control_frame,
            text="开始训练",
            command=self.start_training,
            width=15
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = tk.Button(
            self.control_frame,
            text="暂停",
            command=self.pause_training,
            width=15,
            state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(
            self.control_frame,
            text="重置",
            command=self.reset_training,
            width=15
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # 状态信息
        self.status_frame = tk.Frame(self.right_frame)
        self.status_frame.pack(fill=tk.X, pady=5)

        self.episode_var = tk.StringVar(value="回合: 0")
        self.episode_label = tk.Label(self.status_frame, textvariable=self.episode_var)
        self.episode_label.pack(side=tk.LEFT, padx=5)

        self.step_var = tk.StringVar(value="步数: 0")
        self.step_label = tk.Label(self.status_frame, textvariable=self.step_var)
        self.step_label.pack(side=tk.LEFT, padx=5)

        self.reward_var = tk.StringVar(value="奖励: 0")
        self.reward_label = tk.Label(self.status_frame, textvariable=self.reward_var)
        self.reward_label.pack(side=tk.LEFT, padx=5)

        # Q表显示
        self.q_table_frame = tk.Frame(self.right_frame)
        self.q_table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.q_table_label = tk.Label(self.q_table_frame, text="Q表:", font=("Arial", 12, "bold"))
        self.q_table_label.pack(anchor=tk.W)

        # 创建Q表显示区域
        self.create_q_table_display()

        # 进度条
        self.progress_frame = tk.Frame(self.right_frame)
        self.progress_frame.pack(fill=tk.X, pady=10)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=MAX_EPISODE
        )
        self.progress_bar.pack(fill=tk.X)

    def create_q_table_display(self):
        """创建Q表显示区域"""
        self.q_tree = ttk.Treeview(self.q_table_frame, columns=["state"] + ACTIONS, show="headings")

        # 设置列标题
        self.q_tree.heading("state", text="状态")
        self.q_tree.heading("UP", text="上")
        self.q_tree.heading("DOWN", text="下")
        self.q_tree.heading("LEFT", text="左")
        self.q_tree.heading("RIGHT", text="右")

        # 设置列宽
        self.q_tree.column("state", width=80)
        self.q_tree.column("UP", width=60)
        self.q_tree.column("DOWN", width=60)
        self.q_tree.column("LEFT", width=60)
        self.q_tree.column("RIGHT", width=60)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.q_table_frame, orient="vertical", command=self.q_tree.yview)
        self.q_tree.configure(yscroll=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.q_tree.pack(fill=tk.BOTH, expand=True)

    def update_grid_visualization(self):
        """更新网格可视化"""
        self.grid_canvas.delete("all")

        # 绘制网格
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1, y1 = j * CELL_SIZE, i * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

                # 目标位置特殊颜色
                if (i, j) == TERMINAL_POS:
                    self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill="lightgreen", outline="black")
                    self.grid_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="目标", font=("Arial", 12))
                else:
                    self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")

                # 显示坐标
                self.grid_canvas.create_text(x1 + 15, y1 + 15, text=f"({j},{i})", font=("Arial", 8))

        # 绘制智能体
        if not self.terminated:
            x, y = self.state
            x1, y1 = x * CELL_SIZE, y * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

            # 绘制圆形表示智能体
            padding = 10
            self.grid_canvas.create_oval(
                x1 + padding, y1 + padding,
                x2 - padding, y2 - padding,
                fill="blue"
            )
            self.grid_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="signal", fill="white", font=("Arial", 10))

        # 显示历史路径
        if self.history:
            for i in range(len(self.history) - 1):
                x1, y1 = self.history[i]
                x2, y2 = self.history[i + 1]

                # 转换为画布坐标
                canvas_x1 = x1 * CELL_SIZE + CELL_SIZE / 2
                canvas_y1 = y1 * CELL_SIZE + CELL_SIZE / 2
                canvas_x2 = x2 * CELL_SIZE + CELL_SIZE / 2
                canvas_y2 = y2 * CELL_SIZE + CELL_SIZE / 2

                self.grid_canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2, fill="red", width=2)

    def update_q_table_display(self):
        """更新Q表显示"""
        # 清空现有内容
        for item in self.q_tree.get_children():
            self.q_tree.delete(item)

        # 添加Q表内容
        for state in self.q_table.index:
            x, y = state
            state_str = f"({x},{y})"
            values = [state_str] + [f"{val:.2f}" for val in self.q_table.loc[state].values]
            self.q_tree.insert("", "end", values=values)

    def start_training(self):
        """开始训练"""
        if not self.is_training:
            self.is_training = True
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.DISABLED)

            # 如果是新回合，随机初始化状态
            if self.episode == 0 or self.terminated:
                self.state = (np.random.randint(0, GRID_SIZE - 1), np.random.randint(0, GRID_SIZE - 1))
                self.terminated = False
                self.step = 0
                self.history = [self.state]
                self.episode_var.set(f"回合: {self.episode + 1}")

            # 开始训练循环
            self.train_step()

    def train_step(self):
        """执行一步训练（改进版：自动连续训练）"""
        if not self.is_training:
            return

        # 执行当前步骤
        action = self.choose_action(self.state)
        next_state, reward = self.get_env_back(self.state, action)

        # 更新Q表
        if next_state != TERMINAL_POS:
            q_target = reward + GAMMA * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward
            self.terminated = True

        q_predict = self.q_table.loc[self.state, action]
        self.q_table.loc[self.state, action] += ALPHA * (q_target - q_predict)

        # 更新状态和历史记录
        self.state = next_state
        self.step += 1
        self.history.append(next_state)

        # 更新UI
        self.step_var.set(f"步数: {self.step}")
        self.reward_var.set(f"奖励: {reward}")
        self.update_grid_visualization()
        self.update_q_table_display()

        # 检查是否完成当前episode
        if self.terminated:
            self.episode += 1
            self.progress_var.set(self.episode)

            # 检查是否完成所有回合
            if self.episode >= MAX_EPISODE:
                self.is_training = False
                self.start_btn.config(state=tk.DISABLED)
                self.pause_btn.config(state=tk.DISABLED)
                self.reset_btn.config(state=tk.NORMAL)
                messagebox.showinfo("训练完成", "所有回合训练已完成！")
                return

            # 自动开始下一个episode
            self.state = (np.random.randint(0, GRID_SIZE - 1), np.random.randint(0, GRID_SIZE - 1))
            self.terminated = False
            self.step = 0
            self.history = [self.state]
            self.episode_var.set(f"回合: {self.episode + 1}")

        # 安排下一步训练（控制速度）
        self.root.after(500, self.train_step)

    def pause_training(self):
        """暂停训练"""
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)

    def reset_training(self):
        """重置训练"""
        self.pause_training()
        self.episode = 0
        self.state = (0, 0)
        self.terminated = False
        self.step = 0
        self.history = []
        self.q_table = self.build_table(GRID_SIZE, ACTIONS)

        self.episode_var.set(f"回合: {self.episode}")
        self.step_var.set(f"步数: {self.step}")
        self.reward_var.set(f"奖励: 0")
        self.progress_var.set(0)

        self.update_grid_visualization()
        self.update_q_table_display()
        self.start_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = QLearningGUI(root)
    root.mainloop()