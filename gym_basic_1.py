import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")


episodes = 10
for _ in range(episodes):
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            print("Episode finished after {} timesteps".format(_))
            break
        env.render()

env.close()