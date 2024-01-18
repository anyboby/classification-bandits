from mnist import MNISTBandit
from fashionmnist import FashionMNISTBandit
from bsuite.utils import gym_wrapper

env = MNISTBandit()
gym_env = gym_wrapper.GymFromDMEnv(env)
gym_env.reset()

for i in range(1000):
    _, rew, done, info = gym_env.step(gym_env.action_space.sample())
    if i%25==0:
        print("reward: ", rew)
    if done:
        obs = gym_env.reset()


env = FashionMNISTBandit()
gym_env = gym_wrapper.GymFromDMEnv(env)
gym_env.reset()

for i in range(1000):
    _, rew, done, info = gym_env.step(gym_env.action_space.sample())
    if i%25==0:
        print("reward: ", rew)
    if done:
        obs = gym_env.reset()