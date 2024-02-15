from piiw.utils.utils import random_index
import torch
import numpy as np


class ExperienceReplay:
    def __init__(self, keys, capacity):
        self._keys = keys
        self._capacity = capacity
        self._data = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data)

    def keys(self):
        return iter(self._keys)

    def append(self, example):
        if isinstance(example, dict):
            example = [example[k] for k in self._keys]
        assert len(example) == len(self._keys)

        idx = self._next_idx
        if self._next_idx >= len(self._data):
            self._data.append(example)
        else:
            self._data[self._next_idx] = example

        self._next_idx = self._next_idx + 1
        if self._next_idx == self._capacity:
            self._next_idx = 0
        return idx

    def get_batch(self, indices):
        datapoints = [self._data[i] for i in indices]
        batch = [torch.tensor(np.array(l)) for l in zip(*datapoints)]
        return {k: b for k, b in zip(self._keys, batch)}

    def sample(self, size):
        indices = random_index(len(self), size, replace=False)
        return indices, self.get_batch(indices)

    def sample_one(self):
        return self._data[np.random.choice(len(self._data))]
