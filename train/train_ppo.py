import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("ma_gym:HuRoSorting-v0", n_envs=4)
# env = gym.make("ma_gym:HuRoSorting-v0")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000000)
model.save("ppo_hurosorting")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_hurosorting")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
