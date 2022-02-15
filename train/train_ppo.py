from tabnanny import verbose
import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("ma_gym:HuRoSorting-v0", n_envs=4)
# env = gym.make("ma_gym:HuRoSorting-v0")

if os.path.exists("results/ppo_hurosorting/model_dump.zip"):
    model = PPO.load("results/ppo_hurosorting/model_dump.zip", env,  verbose=1)
else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/progress_tensorboard/", gamma=0.8, device='cpu')
    model.learn(total_timesteps=500000)
    model.save("results/ppo_hurosorting/model_dump.zip")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# del model # remove to demonstrate saving and loading