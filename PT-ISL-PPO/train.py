import MyEnv
import gymnasium as gym
from stable_baselines3 import PPO


env = MyEnv.GridWorldEnv(grid_size=10)

obs, info = env.reset()
env.render()

# 进行一个随机的回合（episode）作为演示
terminated = False
total_reward = 0
steps = 0

while not terminated:
    # 从动作空间中随机选择一个动作
    action = env.action_space.sample() 
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 累加奖励和步数
    total_reward += reward
    steps += 1
    
    # 渲染环境
    print(f"Step: {steps}, Action: {action}, Reward: {reward:.2f}")
    env.render()
    
    if terminated or truncated:
        print(f"Episode finished after {steps} steps with total reward: {total_reward:.2f}")
        break

env.close()