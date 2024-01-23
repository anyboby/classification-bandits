"""MNIST classification as a bandit.

Adapted from https://github.com/google-deepmind/bsuite/tree/main/bsuite
"""

from bsuite.environments import base
from bsuite.experiments.mnist import sweep

import dm_env
from dm_env import specs
import numpy as np

import array
import gzip
import os
from os import path
import struct

import urllib.request

_DATA = "/tmp/jax_example_data/"

_LABEL_LEGEND = {0 : "0",
                 1 : "1",
                 2 : "2",
                 3 : "3",
                 4 : "4",
                 5 : "5",
                 6 : "6",
                 7 : "7",
                 8 : "8",
                 9 : "9",}

def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print(f"downloaded {url} to {_DATA}")

def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels


def load_mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = train_images / np.float32(255.)
  test_images = test_images / np.float32(255.)
  
  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return (train_images, train_labels), (test_images, test_labels)

class MNISTBandit(base.Environment):
  """MNIST classification as a bandit environment."""

  def __init__(self, fraction: float = 1., seed: int = None):
    """Loads the MNIST training set (60K images & labels) as numpy arrays.

    Args:
      fraction: What fraction of the training set to keep (default is all).
      seed: Optional integer. Seed for numpy's random number generator (RNG).
    """
    super().__init__()
    (images, labels), _ = load_mnist()

    num_data = len(labels)

    self._num_data = int(fraction * num_data)
    self._image_shape = images.shape[1:]

    self._images = images[:self._num_data]
    self._labels = labels[:self._num_data]
    self._rng = np.random.RandomState(seed)
    self._correct_label = None

    self._total_regret = 0.
    self._optimal_return = 1.
    self._label_legend = _LABEL_LEGEND

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _reset(self) -> dm_env.TimeStep:
    """Agent gets an MNIST image to 'classify' using its next action."""
    idx = self._rng.randint(self._num_data)
    image = self._images[idx].astype(np.float32) / 255
    self._correct_label = self._labels[idx]

    return dm_env.restart(observation=image)

  def _step(self, action: int) -> dm_env.TimeStep:
    """+1/-1 for correct/incorrect guesses. This also terminates the episode."""
    correct = action == self._correct_label
    reward = 1. if correct else -1.
    self._total_regret += self._optimal_return - reward
    observation = np.zeros(shape=self._image_shape, dtype=np.float32)
    return dm_env.termination(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=self._image_shape, dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(num_values=10)

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)
