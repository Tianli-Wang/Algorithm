# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.registration import register

# 注册环境
register(
    id='NonUniformGridWorld-v0',
    entry_point='MyEnv:NonUniformGridWorldEnv',
    kwargs={'grid_size': 10, 'obstacle_ratio': 0.15}
)

# 创建训练环境（无渲染）
env = gym.make('NonUniformGridWorld-v0')
check_env(env, warn=True)

# 创建 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    n_steps=128,
    batch_size=64,
    n_epochs=4,
    learning_rate=0.001,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
)

# 训练
print("开始训练...")
model.learn(total_timesteps=100_000)
model.save("ppo_gridworld")
env.close()

print("模型已保存为 'ppo_gridworld.zip'")