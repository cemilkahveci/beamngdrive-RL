from stable_baselines3 import PPO
import os
from ObstacleCar import ObstacleEnv
import time




logdir = "logs/PPO_0"


models_dir = "models/PPO"
env = ObstacleEnv()
env.reset()

model_path = f"{models_dir}/1020000.zip"
model = PPO.load(model_path, env=env)

TIMESTEPS = 30000
iters = 34
while True:
	iters += 1

	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")