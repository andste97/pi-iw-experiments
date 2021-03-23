from collections import defaultdict
from utils import sample_pmf, softmax
import numpy as np
from widthbased_planner import RolloutWidthPlanner


class CountbasedRolloutIW(RolloutWidthPlanner):
    def __init__(self, generate_successor_fn, width, ignore_cached_nodes=False, ignore_terminal_nodes=False, temp=0.005, features_name="features", ensure_same_initialization=False):
        RolloutWidthPlanner.__init__(self, generate_successor_fn=generate_successor_fn,
                                     width=width,
                                     ignore_cached_nodes=ignore_cached_nodes,
                                     ignore_terminal_nodes=ignore_terminal_nodes,
                                     features_name=features_name)
        self.features_name = features_name
        self.reset_counts()
        self.temp = temp
        self.ensure_same_initialization = ensure_same_initialization

    def reset_counts(self):
        self.visits = defaultdict(lambda: 0)

    def initialize(self, tree):
        novelty_table = self.create_novelty_table()

        assert self.features_name in tree.root.data.keys(), "IW planners require the state to be factored into features"

        nodes_to_check = list()
        unpruned = dict()
        for node in tree.iter_insertion_order():
            old_solved_label = node.solved if hasattr(node, "solved") else False
            node.in_queue = False
            if self.check_update_novelty(node, novelty_table, caching=True):
                if node.data[self.features_name] in unpruned:
                    self.prune(unpruned, unpruned[node.data[self.features_name]])
                unpruned[node.data[self.features_name]] = node
                node.in_queue = True

            if self.ensure_same_initialization:
                if old_solved_label and not node.solved:
                    nodes_to_check.append(node)

        if self.ensure_same_initialization:
            for node in nodes_to_check:
                novel = novelty_table.check(node.data[self.features_name], node.depth, node_is_new=False)
                if not novel:
                    self.prune(unpruned, node)

        tree.root.unpruned = unpruned
        tree.root.expaned_unpruned = dict()
        tree.root.novelty_table = novelty_table

    def plan(self, tree):
        """
        :param tree: Tree to begin expanding nodes. It can contain just the root node (for offline planning or online
        planning without caching nodes), or an many (cached) nodes.
        """
        assert hasattr(tree.root, "novelty_table"), "Planner not initialized."
        self._novelty_table = tree.root.novelty_table
        unpruned = tree.root.unpruned
        expanded_unpruned = tree.root.expaned_unpruned

        while not self.should_stop(tree) and len(unpruned)>0:
            node = self.select(unpruned)

            # Check novelty before generating successor
            features = node.data[self.features_name]
            novel = self._novelty_table.check(features, depth=node.depth)
            if not novel:
                self.prune(unpruned, node)
                continue

            self.rollout(tree, node, unpruned, expanded_unpruned)

    def select(self, unpruned):
        # Get all feature values in the tree
        reachable_features = list(unpruned.keys())
        # Select state based on counts
        counts = np.array([self.visits[f] for f in reachable_features])
        probs = softmax(1 / (counts + 1), temp=self.temp)
        features = reachable_features[sample_pmf(probs)]
        return unpruned[features]

    def rollout(self, tree, node, unpruned, expanded_unpruned):
        while not self.should_stop(tree):
            assert node is not None
            self.visits[node.data[self.features_name]] += 1
            child = self.generate_successor(tree, node)

            if child is None:
                # If should stop is true, the node may not be expanded yet so don't remove it!
                if not self.should_stop(tree):
                    # No more children from this node, it has been expanded, remove it from the list
                    self.remove_node(unpruned, node)
                    expanded_unpruned[node.data[self.features_name]] = node
                break

            child.in_queue = False
            if not self.check_update_novelty(child, self._novelty_table, caching=False):
                break

            # Prune other node with same features, in case there is, together with its descendants
            n = None
            features = child.data[self.features_name]
            if features in unpruned:
                n = unpruned[features]
            if features in expanded_unpruned:
                n = expanded_unpruned[features]
            if n is not None:
                self.prune(unpruned, n)

            # Add to unpruned mapping (features -> child)
            unpruned[child.data[self.features_name]] = child
            child.in_queue = True
            node = child

    def prune(self, unpruned, node):
        for n in node.breadth_first():
            if n.in_queue:
                self.remove_node(unpruned, n)

    def remove_node(self, unpruned, node):
        assert node.in_queue
        features = node.data[self.features_name]
        unpruned.pop(features)
        node.in_queue = False
