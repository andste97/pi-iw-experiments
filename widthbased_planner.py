from collections import defaultdict
from itertools import combinations
import numpy as np
from planner import Planner

class WidthPlanner(Planner):
    def __init__(self, generate_successor_fn, width, ignore_cached_nodes, ignore_terminal_nodes, features_name):
        super(WidthPlanner, self).__init__(generate_successor_fn)
        self.width = width
        self.ignore_cached_nodes = ignore_cached_nodes
        self.ignore_terminal_nodes = ignore_terminal_nodes
        self.features_name = features_name

    def check_update_novelty(self, node, novelty_table, caching):
        """"Returns true if the node is novel and should not be pruned, if node is not novel,
        will return false and set node.pruned to true."""
        node.pruned = False

        if caching and self.ignore_cached_nodes:
            if not node.data["done"] and not self.should_prune(node):
                return True
        else:
            if node.data["done"]:
                if not self.ignore_terminal_nodes:
                    # add terminal node's features to novelty table
                    _ = self.check_update_novelty_table(node, novelty_table)
            else:
                # add cached nodes to novelty table, and maybe prune them
                novel = self.check_update_novelty_table(node, novelty_table)
                if novel:
                    if self.should_prune(node):
                        return False
                    return True
                else:
                    node.pruned = True
                    return False
        return False

    def check_update_novelty_table(self, node, novelty_table):
        raise NotImplementedError()


class BFSWidthPlanner(WidthPlanner):
    def __init__(self, generate_successor_fn, width, ignore_cached_nodes, ignore_terminal_nodes, features_name):
        super(BFSWidthPlanner, self).__init__(generate_successor_fn, width, ignore_cached_nodes,
                                              ignore_terminal_nodes, features_name)

    def create_novelty_table(self):
        if self.width is None:
            return NoveltyNTable()
        elif self.width == 1:
            return Novelty1Table()  # more efficient?
        else:
            return BinaryNoveltyTable(self.width, Novelty1Table)

    def check_update_novelty_table(self, node, novelty_table):
        return novelty_table.check_and_update(node.data[self.features_name])


class RolloutWidthPlanner(WidthPlanner):
    def __init__(self, generate_successor_fn, width, ignore_cached_nodes, ignore_terminal_nodes, features_name,
                 n_features=None, n_values=None):
        self.n_features = n_features
        self.n_values = n_values
        assert (self.n_features is None and self.n_values is None) or (self.n_features is not None and self.n_values is not None)
        super(RolloutWidthPlanner, self).__init__(generate_successor_fn, width, ignore_cached_nodes,
                                                  ignore_terminal_nodes, features_name)

    def create_novelty_table(self):
        if self.width is None:
            return DepthNoveltyNTable()
        elif self.width == 1:
            if self.n_features is not None:
                return DepthFeaturesNovelty1Table(self.n_features, self.n_values)
            else:
                return DepthAtomsNovelty1Table()  # more efficient?
        else:
            return BinaryNoveltyTable(self.width, DepthAtomsNovelty1Table)

    def check_update_novelty_table(self, node, novelty_table):
        return novelty_table.check_and_update(node.data[self.features_name], node.depth)

    def recheck_novelty(self, node, novelty_table):
        return novelty_table.check(node.data[self.features_name], node.depth)


class RolloutLabelsWidthPlanner(RolloutWidthPlanner):
    def __init__(self, generate_successor_fn, width, ignore_cached_nodes, ignore_terminal_nodes, features_name, branching_factor,
                 n_features=None, n_values=None):
        super(RolloutLabelsWidthPlanner, self).__init__(generate_successor_fn, width, ignore_cached_nodes,
                                                        ignore_terminal_nodes, features_name, n_features, n_values)
        self.branching_factor = branching_factor

    def maybe_solve(self, node, novelty_table, caching):
        node.solved = False
        if not self.check_update_novelty(node, novelty_table, caching):
            self.solve_and_propagate_label(node)

    def solve_and_propagate_label(self, node):
        node.solved = True
        for n in node.ascendants():
            if self.check_all_children_solved(n):
                n.solved = True
            else:
                break

    def check_all_children_solved(self, node):
        if len(node.children) == self.branching_factor and all(child.solved for child in node.children):
            assert len(set([child.data["a"] for child in node.children])) == self.branching_factor
            return True
        return False


# ---------------
# Novelty tables
# ---------------
class Novelty1Table():
    def __init__(self):
        self.table = set()

    def check(self, atoms):
        return any(atom for atom in atoms if atom not in self.table)

    def check_and_update(self, atoms):
        l = len(self.table)
        self.table.update(atoms)
        return len(self.table) != l


class NoveltyNTable:
    def __init__(self):
        self.visited = set()

    def check(self, atoms):
        if atoms in self.visited:
            return False
        return True

    def check_and_update(self, atoms):
        """
        Evaluates the novelty of a state up to the pre-set max-width.
        """
        if atoms in self.visited:
            return False
        self.visited.add(atoms)
        return True


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


class DepthFeaturesNovelty1Table:
    def __init__(self, n_feats, n_values):
        self.n_feats = n_feats
        self.n_values = n_values
        self.features_indices = np.arange(n_feats)
        self.features_depth = np.full(shape=(n_feats, n_values), fill_value=np.inf)

    def check(self, feature_values, depth):
        return np.any(depth <= self.features_depth[self.features_indices, feature_values])

    def check_and_update(self, feature_values, depth):
        assert len(feature_values) == self.n_feats
        mask = depth < self.features_depth[self.features_indices, feature_values]
        if np.any(mask):
            self.features_depth[self.features_indices[mask], np.asarray(feature_values)[mask]] = depth  # TODO: mask only once, keep features_depth[idx, values] as a variable and reuse it
            return True
        return False


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


class BinaryNoveltyTable:
    # Directly checks width=max_width, returns True/False
    def __init__(self, max_width, novelty1_table_cls):
        self.max_width = max_width
        self.table = novelty1_table_cls()

    def check(self, atoms, *args, **kwargs):
        # Iterate for each value of k, and process all tuples of size k to check for novel ones.
        if self.table.check([frozenset(c) for c in combinations(atoms, self.max_width)],  *args, **kwargs) == 1:
            return True
        return False

    def check_and_update(self, atoms, *args, **kwargs):
        """
        Evaluates the novelty of a state up to the pre-set max-width.
        Note that even if we find that a state has novelty e.g. 1, we still iterate through all tuples
        of larger sizes so that they can be recorded in the novelty tables.
        """
        if self.table.check_and_update([frozenset(c) for c in combinations(atoms, self.max_width)],  *args, **kwargs) == 1:
            return True
        return False

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