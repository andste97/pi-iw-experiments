from collections import defaultdict
from itertools import combinations

import numpy as np


class NoveltyTable:
    # Checks widths i=1, ... max_width, returns i or np.inf
    def __init__(self, max_width, novelty1_table_cls):
        self.max_width = max_width
        # We'll have one novelty table for each width value; for instance, tables[2] will contain all
        # tuples of size 2 that have been seen in the search so far.
        self.tables = defaultdict(novelty1_table_cls)

    def check(self, atoms):
        # Iterate for each value of k, and process all tuples of size k to check for novel ones.
        for k in range(1, self.max_width + 1):
            if self.tables[k].check(combinations(atoms, k)):
                return k
        return np.inf

    def check_and_update(self, atoms):
        """
        Evaluates the novelty of a state up to the pre-set max-width.
         that even if we find that a state has novelty e.g. 1, we still iterate through all tuples
        of larger sizes so that they can be recorded in the novelty tables.
        """
        novelty = np.inf
        for k in range(1, self.max_width + 1):
            if self.tables[k].check_and_update(combinations(atoms, k)):
                novelty = min(novelty, k)
        return novelty
