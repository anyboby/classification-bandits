"""Fashion MNIST classification as a bandit.

Adapted from https://github.com/google-deepmind/bsuite/tree/main/bsuite
"""

from bsuite.environments import base
from bsuite.experiments.mnist import sweep

import dm_env
from dm_env import specs
import numpy as np

_LABEL_LEGEND = {0 : "Left",
                 1 : "Right",}

class OneDBandit(base.Environment):
  """1-Dimensional context bandit with 2 arms."""

  def __init__(self, s_min=-1.0, s_max=1.0, n_data=4, seed=0, fraction=1.0):
    """
    """
    super().__init__()

    self._label_legend = _LABEL_LEGEND
    self._num_data = n_data

    self._states = np.linspace(s_min, s_max, n_data)[...,None]
    self._stateshape = (1,)
    self._labels = np.random.randint(0,1, size=(self._num_data, 1))

    self._rng = np.random.RandomState(seed)
    self._correct_label = None

    self._label_legend = _LABEL_LEGEND
    
    self._total_regret = 0.
    self._optimal_return = 1.

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _reset(self) -> dm_env.TimeStep:
    """Agent gets an MNIST image to 'classify' using its next action."""
    idx = self._rng.randint(self._num_data)
    state = self._states[idx].astype(np.float32)
    self._correct_label = self._labels[idx]

    return dm_env.restart(observation=state)

  def _step(self, action: int) -> dm_env.TimeStep:
    """+0.1/-0.1 for correct/incorrect guesses. This also terminates the episode."""
    correct = action == self._correct_label
    reward = 0.01 if correct else -0.01
    self._total_regret += self._optimal_return - reward
    observation = np.zeros(shape=self._stateshape, dtype=np.float32)
    return dm_env.termination(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=self._stateshape, dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(num_values=2)

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)
