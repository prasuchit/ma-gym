from tabnanny import verbose
import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("ma_gym:HuRoSorting-v0", n_envs=4)
# env = gym.make("ma_gym:HuRoSorting-v0")

test = True

if os.path.exists("results/ppo_hurosorting/model_dump.zip") and test:
    model = PPO.load("results/ppo_hurosorting/model_dump.zip", env,  verbose=1)
else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/progress_tensorboard/", device='cpu')
    model.learn(total_timesteps=500000)
    model.save("results/ppo_hurosorting/model_dump.zip")

env = gym.make('ma_gym:HuRoSorting-v0')
obs = env.reset()
dones = False

while not dones:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action, verbose=1)
    print(f'reward: {rewards}')
    # env.render()

# del model # remove to demonstrate saving and loading