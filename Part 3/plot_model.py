from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from matplotlib import pyplot as plt
import pprint
import numpy as np
import os

env = gym.make("intersection-v0", render_mode="rgb_array")
env.reset()

model_path = "example_model.zip"
model = DQN.load(model_path, env=env, custom_objects = {'observation_space': env.observation_space, 'action_space': env.action_space})

print(model.num_timesteps)

episodes = 10

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    while not (done or truncated):
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()
        total_reward += rewards
        
    
    print(f"Episode {ep+1} final reward is: {total_reward}")


