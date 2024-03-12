"""Fashion MNIST classification as a bandit.

Adapted from https://github.com/google-deepmind/bsuite/tree/main/bsuite
"""

from bsuite.environments import base
from bsuite.experiments.mnist import sweep

import dm_env
from dm_env import specs
import numpy as np

from torchvision import datasets, transforms
import torch
import PIL.Image

import array
import gzip
import os
from os import path
import struct

import urllib.request


class ToArray(torch.nn.Module):
    '''convert image to float and 0-1 range'''
    dtype = np.float32

    def __call__(self, x):
        assert isinstance(x, PIL.Image.Image)
        x = np.asarray(x, dtype=self.dtype)
        x /= 255.0
        return x

_DATA = "/tmp/jax_example_data_cifar10/"

_LABEL_LEGEND = {0 : "T-shirt/top",
                 1 : "Trouser",
                 2 : "Pullover",
                 3 : "Dress",
                 4 : "Coat",
                 5 : "Sandal",
                 6 : "Shirt",
                 7 : "Sneaker",
                 8 : "Bag",
                 9 : "Ankleboot",}

def cifar10_raw():
  """Download and parse the raw CIFAR10 dataset."""
  img_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    ToArray(),
  ])

  train_dataset = datasets.CIFAR10(_DATA, train=True, download=True, transform=img_transforms)
  test_dataset = datasets.CIFAR10(_DATA, train=False, transform=img_transforms)

  return train_dataset, test_dataset

def load_cifar10(permute_train=False):
  """Download, parse and process Fashion MNIST data to unit scale and one-hot labels."""
  train_dataset, test_dataset = cifar10_raw()  
  train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset))
  test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

  train_images, train_labels = next(iter(train_dataloader))
  test_images, test_labels = next(iter(test_dataloader))
  
  train_images, train_labels = train_images.numpy(), train_labels.numpy()
  test_images, test_labels = test_images.numpy(), test_labels.numpy()

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]
  
  infos = {"label_legend": {i:train_dataset.classes[i] for i in range(len(train_dataset.classes))}}
  return (train_images, train_labels), (test_images, test_labels), infos

class CIFAR10Bandit(base.Environment):
  """Fashion MNIST classification as a bandit environment."""

  def __init__(self, fraction: float = 1., seed: int = None):
    """Loads the MNIST training set (60K images & labels) as numpy arrays.

    Args:
      fraction: What fraction of the training set to keep (default is all).
      seed: Optional integer. Seed for numpy's random number generator (RNG).
    """
    super().__init__()
    (images, labels), _, infos = load_cifar10()

    self._label_legend = infos["label_legend"]
    num_data = len(labels)

    self._num_data = int(fraction * num_data)
    self._image_shape = images.shape[1:]

    self._images = images[:self._num_data]
    self._labels = labels[:self._num_data]
    self._rng = np.random.RandomState(seed)
    self._correct_label = None

    self._label_legend = _LABEL_LEGEND
    
    self._total_regret = 0.
    self._optimal_return = 1.

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
