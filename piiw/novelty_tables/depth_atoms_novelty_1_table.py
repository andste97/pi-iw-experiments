from collections import defaultdict

import numpy as np


class DepthAtomsNovelty1Table:
    def __init__(self):
        self.atom_depth = defaultdict(lambda: np.inf)

    def check(self, atoms, depth):
        for atom in atoms:
            if depth <= self.atom_depth[atom]:
                return True  # at least one atom is either case 1 or 4
        return False  # all atoms are either case 2 or 3
        # return any(depth <= self.atom.depth[atom] for atom in atoms)

    def check_and_update(self, atoms, depth):
        is_novel = False
        for atom in atoms:
            if depth < self.atom_depth[atom]:
                self.atom_depth[atom] = depth
                is_novel = True  # case 1, novel
        return is_novel
