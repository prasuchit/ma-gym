import gym
import pytest


@pytest.fixture(scope='module')
def env():
    env = gym.make('ma_gym:HuRoSorting-v0')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    # assert env._total_episode_reward == [0 for _ in range(env.n_agents)]
    assert env._agent_dones == False


def test_reset_after_episode_end(env):
    env.reset()
    done = False
    step_i = 0
    ep_reward = 0
    while not done:
        step_i += 1
        _, reward_n, done = env.step(env.action_space.sample())
        print(f"Here's the reward I got: {reward_n} at step: {step_i} and done is {done}")
        ep_reward += reward_n

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


if __name__ == "__main__":
    env = gym.make("ma_gym:HuRoSorting-v0")
    test_reset_after_episode_end(env)
