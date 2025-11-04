# demo.py
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO

GLOBAL_SEED=42

# 导入并注册环境
from MyEnv import NonUniformGridWorldEnv
gym.register(
    id="NonUniformGridWorld-v0",
    entry_point=NonUniformGridWorldEnv,
    kwargs={"grid_size": 10, "obstacle_ratio": 0.15, 'seed': GLOBAL_SEED}
)

# 加载模型
model = PPO.load("ppo_gridworld_final")

# 创建可渲染环境
env = gym.make("NonUniformGridWorld-v0", render_mode="console")

# 获取原始环境（用于 monkey-patch render，确保清屏生效）
raw_env = env
while hasattr(raw_env, 'env') and raw_env.env is not None:
    raw_env = raw_env.env

# 运行演示
print("start to observe the result...")

time.sleep(2)

obs, _ = env.reset()
raw_env.render()
time.sleep(1)

total_reward = 0.0
for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    raw_env.render()  # 清屏刷新
    total_reward += reward
    time.sleep(0.4)
    
    if terminated or truncated:
        print(f"\n成功！总步数: {step + 1}, 总奖励: {total_reward:.2f}")
        break

env.close()