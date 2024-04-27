from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import os


collision_reward = -200 #default -5
arrived_reward = 1 #default 1
high_speed_reward = 1 #default 1

policy = "MlpPolicy"
gamma = 0.6

models_dir = f"models_gamma{int(100*gamma)}_{policy}_col{collision_reward}_arr{arrived_reward}_spd{high_speed_reward}/DQN"
logdir = f"logs_gamma{int(100*gamma)}_{policy}_col{collision_reward}_arr{arrived_reward}_spd{high_speed_reward}"
tb_log_name = "DQN"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gym.make("intersection-v0", render_mode="rgb_array")
env.config["collision_reward"] = collision_reward 
env.config["arrived_reward"] = arrived_reward
env.config["high_speed_reward"] = high_speed_reward
env.reset()

n_cpu = 8
batch_size = 64

model = DQN(policy, env, verbose=2, tensorboard_log=logdir
            , gamma=gamma
            , batch_size=batch_size)

TIMESTEPS = 500
n_iterations = 60
for i in range(n_iterations):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=tb_log_name)
    model.save(f"{models_dir}/{model.num_timesteps}")
    print(f"Iteration {i+1}/{n_iterations} done")
