from mnist import MNISTBandit
from bsuite.utils import gym_wrapper

env = MNISTBandit()
gym_env = gym_wrapper.GymFromDMEnv(env)
gym_env.reset()

for _ in range(1000):
    obs, rew, done, info = gym_env.step(gym_env.action_space.sample())
    print("reward: ", rew)
    if done:
        _ = gym_env.reset()