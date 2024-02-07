from piiw.utils.utils import random_index


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

    def extend(self, examples):
        if isinstance(examples, dict):
            assert set(examples.keys()) == set(self._keys)
            examples = list(zip(*[examples[k] for k in self._keys]))
        assert all(len(l) == len(self._keys) for l in examples)

        if len(examples) > self._capacity:
            examples = examples[-self._capacity:]  # there shouldn't be more examples than buffer capacity

        remaining_slots = self._capacity - self._next_idx  # slots remaining before overflowing
        l1 = examples[:remaining_slots]
        l2 = examples[remaining_slots:]

        if len(self._data) < self._capacity:
            self._data.extend(l1)
        else:
            self._data[self._next_idx:self._next_idx+len(l1)] = l1
        indices = list(range(self._next_idx, self._next_idx + len(l1)))

        if len(l2) > 0:
            self._data[:len(l2)] = l2
            indices += list(range(len(l2)))

        assert len(indices) == len(examples)
        self._next_idx = (self._next_idx + len(examples)) % self._capacity
        return indices

    def get_batch(self, indices):
        datapoints = [self._data[i] for i in indices]
        batch = [list(l) for l in zip(*datapoints)]  # TODO: numpy array instead of list
        return {k: b for k, b in zip(self._keys, batch)}

    def sample(self, size):
        indices = random_index(len(self), size, replace=False)
        return indices, self.get_batch(indices)
