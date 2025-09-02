import gymnasium as gym

env = gym.make('Pendulum-v1')
state, info = env.reset()
step = 0

while True:
    action = env.action_space.sample()  # 随机动作
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1
    print(f"Step {step}: terminated={terminated}, truncated={truncated}, done={done}")
    if done:
        print("Episode ended!")
        break
    if step > 300:  # 安全退出
        print("Safety break")
        break