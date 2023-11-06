# Classification bandits
This is a set of bandit problems based on common classification datasets. Insipired by the bsuite mnist bandit (https://github.com/google-deepmind/bsuite). 

# Usage
Cloning:

```
git clone https://github.com/anyboby/classification-bandits.git
cd classification-bandits
```

The sole requirement is currrently bsuite:

```
pip install bsuite
# or
pip install -r req.txt
```

By default the environments come as DMEnvs but can be easily adapated to gym through the bsuite gymwrapper (see example).

# Example

```
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
```


