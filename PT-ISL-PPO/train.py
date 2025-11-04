# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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

# 使用512个并行环境
n_envs = 512
print(f"使用 {n_envs} 个并行环境进行训练")

# 创建训练环境
env = make_vec_env('NonUniformGridWorld-v0', n_envs=n_envs)

# 创建评估环境
eval_env = make_vec_env('NonUniformGridWorld-v0', n_envs=4)

# 针对大规模并行优化的参数
n_steps = 256               # 适度增加,平衡收集效率
batch_size = 4096           # 大幅增加,适配大buffer
n_epochs = 4                # 保持4轮,避免过拟合
total_timesteps = 50_000_000  # 增加总步数

total_buffer_size = n_steps * n_envs
updates_per_iteration = (total_buffer_size // batch_size) * n_epochs

print(f"\n=== 训练配置 ===")
print(f"n_envs: {n_envs}")
print(f"n_steps: {n_steps} (每个环境)")
print(f"总buffer大小: {total_buffer_size:,} 样本")
print(f"batch_size: {batch_size:,}")
print(f"n_epochs: {n_epochs}")
print(f"\n每次更新:")
print(f"  - Mini-batches per epoch: {total_buffer_size // batch_size}")
print(f"  - 总梯度更新次数: {updates_per_iteration}")
print(f"\n总训练:")
print(f"  - 总步数: {total_timesteps:,}")
print(f"  - 预计更新轮次: {total_timesteps // total_buffer_size:,}")
print(f"  - 预计总梯度步数: {(total_timesteps // total_buffer_size) * updates_per_iteration:,}")

# 创建 PPO 模型 (针对512个环境优化)
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    device="cpu",
    
    # 数据收集参数
    n_steps=256,               # 256步 × 512环境 = 131K样本
    
    # 训练参数 - 关键调整!
    batch_size=4096,           # 大batch_size匹配大buffer
    n_epochs=4,                # 4轮足够,避免过拟合
    
    # 学习率 - 大batch需要调整
    learning_rate=3e-4,        # 可以考虑线性衰减
    
    # 折扣和优势估计
    gamma=0.99,
    gae_lambda=0.95,
    
    # 正则化
    ent_coef=0.01,             # 熵系数
    clip_range=0.2,            # PPO clip
    max_grad_norm=0.5,         # 梯度裁剪
    vf_coef=0.5,               # value function系数
    
    # 网络结构
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],     # actor
            vf=[256, 256]      # critic  
        )
    ),
)

# 设置回调函数
checkpoint_callback = CheckpointCallback(
    save_freq=max(10_000, total_buffer_size),  # 至少每次更新后保存
    save_path='./checkpoints/',
    name_prefix='ppo_gridworld'
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=max(20_000, total_buffer_size * 2),  # 每2次更新评估一次
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

# 训练
print("\n" + "="*50)
print("开始训练...")
print("="*50 + "\n")

model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# 保存最终模型
model.save("ppo_gridworld_final")
env.close()
eval_env.close()

print("\n" + "="*50)
print("训练完成!")
print("="*50)
print("最终模型: 'ppo_gridworld_final.zip'")
print("最佳模型: './best_model/' 目录")