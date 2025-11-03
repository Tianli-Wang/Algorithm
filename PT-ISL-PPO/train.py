# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from MyEnv import NonUniformGridWorldEnv
import os

GLOBAL_SEED = 42

# 注册环境
register(
    id='NonUniformGridWorld-v0',
    entry_point='MyEnv:NonUniformGridWorldEnv',
    kwargs={'grid_size': 10, 'obstacle_ratio': 0.15, 'seed': GLOBAL_SEED}
)

# # 创建训练环境（无渲染）
# env = gym.make('NonUniformGridWorld-v0')
# check_env(env, warn=True)

num_cpu = os.cpu_count()
if num_cpu is None:
    num_cpu = 8  # if failed, default 8
print(f"检测到 {num_cpu} 个CPU核心。将使用 {num_cpu} 个并行环境。")

# create parallel environments
env = make_vec_env('NonUniformGridWorld-v0', n_envs=num_cpu)

# 创建 PPO 模型
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    device="cpu",
    n_steps=2048,
    batch_size=128,
    n_epochs=10, # number of policy updates
    learning_rate=0.001,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
)

# 训练
print("start to train...")
model.learn(total_timesteps=1_000_000)
model.save("ppo_gridworld")
env.close()

print("模型已保存为 'ppo_gridworld.zip'")