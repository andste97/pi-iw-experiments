from collections import defaultdict

import numpy as np


class DepthNoveltyNTable:
    def __init__(self):
        self.atoms_depth = defaultdict(lambda: np.inf)

    def check(self, atoms, depth):
        if depth <= self.atoms_depth[atoms]:
            return True  # at least one atom is either case 1 or 4
        return False  # all atoms are either case 2 or 3

    def check_and_update(self, atoms, depth):
        is_novel = False
        if depth < self.atoms_depth[atoms]:
            self.atoms_depth[atoms] = depth
            is_novel = True
        return is_novel
