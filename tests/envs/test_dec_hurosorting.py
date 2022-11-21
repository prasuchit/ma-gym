import gym
import pytest
from stable_baselines3 import PPO

@pytest.fixture(scope='module')

def env():
    env = gym.make('ma_gym:DecHuRoSorting-v0', custom=False)
    yield env
    env.close()

def test_init(env):
    assert env.n_agents == 2

def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert env._agent_dones == False

def test_reset_after_episode_end(env):
    env.reset()
    # obs = env.reset()
    done = False
    step_i = 0
    ep_reward = 0
    while not done:
        step_i += 1
        # print(obs)
        obs, reward_n, done, _ = env.step(env.action_space.sample(), verbose=1)
        # print(reward_n)
        print(f"Here's the reward I got: {reward_n} at step: {step_i} and done is {done} and obs is {obs}")
        # if reward_n != 0 and reward_n != 1:
        #     print(f"Here's the reward I got: {reward_n} at step: {step_i} and done is {done}")
        # ep_reward += reward_n
    assert step_i == env._step_count
    # assert env._total_episode_reward == ep_reward
    test_reset(env)

def test_observation_space(env):
    obs = env.reset()
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(env.observation_space.sample())

def test_rl_learning(env):

    raise NotImplementedError
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=500_000)

    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action, verbose=True)
    #     env.render()
    #     if done:
    #         obs = env.reset()

    # env.close()

if __name__ == "__main__":
    env = gym.make(id="ma_gym:DecHuRoSorting-v0")
    test_reset_after_episode_end(env)
    # test_rl_learning(env)
