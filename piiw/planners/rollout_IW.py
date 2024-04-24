import numpy as np

from tree_utils.node import Node
from utils.utils import sample_pmf
from planners.rollout_labels_width_planner import RolloutLabelsWidthPlanner


class RolloutIW(RolloutLabelsWidthPlanner):
    def __init__(self, generate_successor_fn, branching_factor, width=1, policy_fn=None, ignore_cached_nodes=False,
                 ignore_terminal_nodes=False, min_cum_prob=0, features_name="features", ensure_same_initialization=False,
                 n_features=None, n_values=None):
        """
        :param branching_factor: Number of possible actions
        :param width: Tuples of features of this length will be considered for novelty check
        :param ignore_cached_nodes: If set to True, nodes already existing in the tree will be ignored and their features will not be added to the novelty table
        :param ignore_terminal_nodes: If set to True, terminal nodes (episode done/game over) will be ignored and their features will not be added to the novelty table
        :param min_cum_prob: After discarding all actions that have been solved, if the sum of probabilities for the remaining actions is less than min_cum_prob, the current node will be pruned (set to solved).
        :param policy_fn: Given a node and the number of possible actions, it returns a policy (probability distribution) that will be used for traversing the tree and for generating new nodes.
        """
        super(RolloutIW, self).__init__(generate_successor_fn=generate_successor_fn,
                                        width=width,
                                        ignore_cached_nodes=ignore_cached_nodes,
                                        ignore_terminal_nodes=ignore_terminal_nodes,
                                        features_name=features_name,
                                        branching_factor=branching_factor,
                                        n_features=n_features, n_values=n_values)
        self.uniform_policy_fn = lambda n: np.full(self.branching_factor, 1 / self.branching_factor)
        self.set_policy_fn(policy_fn)
        self.min_cum_prob = min_cum_prob  # Prune when the cumulative probability of the remaining (not solved) actions is lower than this threshold
        self.ensure_same_initialization = ensure_same_initialization

    def set_policy_fn(self, policy_fn):
        if policy_fn is None:
            policy_fn = self.uniform_policy_fn
        self.policy_fn = policy_fn

    def initialize(self, tree):
        """
        Initializes labels of all nodes of the tree to solved = False. Then, sets solved = True if:
            - The node is terminal
            - The node is not novel (if considering cached nodes for novelty)
            - All successors have been solved
        :param tree:
        :return:
        """

        # Warning: calling initialize actually "unsolves" *all* nodes solved at the select step. Thus, re-initializing
        # will cause more select calls to mark those nodes as solved again. Since select consumes random numbers, the
        # result will be different due to randomness. To avoid this, set ensure_same_initialization to True.

        novelty_table = self.create_novelty_table()

        # To deal with an existing tree (maybe initialize novelty table with existing nodes, etc)
        assert self.features_name in tree.root.data.keys(), "Width-based planners require the state to be factored into features"

        nodes_to_check = list()
        for node in tree.iter_insertion_order():
            old_solved_label = node.solved if hasattr(node, "solved") else False
            self.maybe_solve(node, novelty_table, caching=True)

            # So that we also mark as solved nodes solved at select. This is so that we can call initialize at any point
            # without modifying the end result (i.e. we can compare between runs that call initialize more or less
            # times. Not doing this does not change the algorithm, and results may be better or worse due to randomness.
            if self.ensure_same_initialization:
                if old_solved_label and not node.solved:
                    nodes_to_check.append(node)

        if self.ensure_same_initialization:
            for node in nodes_to_check:  # TODO: do it in reverse order
                novel = novelty_table.check(node.data[self.features_name], node.depth)
                if not novel:
                    node.pruned = True
                    self.solve_and_propagate_label(node)

        tree.root.novelty_table = novelty_table
        tree.root.next_rollout_random = False

    def plan(self, tree):
        """
        :param tree: Tree to begin expanding nodes. It can contain just the root node (for offline planning or online
        planning without caching nodes), or an many (cached) nodes.
        """
        assert hasattr(tree.root, "novelty_table"), "Planner not initialized."
        novelty_table = tree.root.novelty_table

        while not self.should_stop(tree) and not tree.root.solved:  # TODO: avoid checking condition twice (here and Rollout while loop)
            node, a = self.select(tree.root, novelty_table, random=tree.root.next_rollout_random)
            if a is not None:
                assert node is not None
                self.rollout(tree, node, a, novelty_table, random=tree.root.next_rollout_random)

    def select(self, node: Node, novelty_table, random):
        """
        Traverses the tree from node on and selects a node and an action that have not yet been expanded.
        :param node: Node where the tree traversing starts from.
        :return: (node, action) to expand, or (None, None) if the subtree has been solved.
        """
        while True:
            assert not node.solved and not node.data["done"], "Solved: %s.  Done: %s.  Depth: %s" % (
            str(node.solved), str(node.data["done"]), str(node.depth))
            novel = novelty_table.check(node.data[self.features_name], node.depth)
            if not novel:
                node.pruned = True
                self.solve_and_propagate_label(node)
                return None, None  # Prune node

            a, child = self.select_action_following_policy(node, random=random)
            assert child is None or (
                        not child.solved and not child.data["done"]), "Solved: %s.  Done: %s.  Depth: %s" % (
            str(child.solved), str(child.data["done"]), str(child.depth))

            if a is None:
                return None, None  # All actions recommended by the policy have been already solved for this node
            else:
                assert a < self.branching_factor, "Ilegal action recommended by the policy (action a=%i >= action_space.n=%i)." % (
                a, self.branching_factor)
                if child is None:
                    return node, a  # Action a needs to be expanded for this node
                else:
                    node = child  # Continue traversing the tree

    def select_action_following_policy(self, node: Node, random):
        """
        Selects an action according to the policy given by _get_policy() (default is uniform distribution). It only
        takes into account nodes that have not been solved yet: it sets probabilities of already solved nodes to 0 and
        samples an action from the normalized resulting policy. It returns:
            - (action, None): if the successor corresponding to the selected action is not in the tree
            - (action, successor): if the successor corresponding to the selected action exists in the tree
            - (None, None): if all actions have been solved (or the sum of probabilities of the remaining actions is
            lower than min_prob) and therefore the current node needs to be pruned
        :param node:
        :return: A tuple (action, successor), (action, None) or (None, None).
        """
        if random:
            policy = self.uniform_policy_fn(node)
        else:
            policy = self.policy_fn(node)

        if node.is_leaf():
            # return action to expand
            assert not node.solved and not node.data["done"], "Solved: %s.  Done: %s.  Depth: %s" % (
            str(node.solved), str(node.data["done"]), str(node.depth))
            return sample_pmf(policy), None

        node_children = [None] * self.branching_factor
        available_actions = (policy > 0)
        for child in node.children:
            node_children[child.data["a"]] = child
            if child.solved:
                available_actions[child.data["a"]] = False

        # Take out actions that have been solved
        p = (policy * available_actions)
        ps = p.sum()

        # No actions available?
        if ps <= self.min_cum_prob:
            # All actions recommended by the policy (i.e. with prob >0) have been (or should be considered) solved. Solve node.
            # It is posible that some nodes in the subtree are not marked as solved. That's not a problem, and it's because the policy gives those branches low probability (less than min_prob)
            self.solve_and_propagate_label(node)
            return None, None

        # Select action not solved
        p = p / ps
        assert all((p > 0) == available_actions), "p: %s;  avail_a: %s;   ps:%s" % (
        str(p), str(available_actions), str(ps))
        a = sample_pmf(p)

        child = node_children[a]
        if child:
            assert not child.solved and not child.data[
                "done"], "a: %i, Solved: %s.  Done: %s.  Depth: %s.  policy: %s.  avail_actions: %s.  p: %s.  ps: %s.  children: %s." % (
            a, str(child.solved), str(child.data["done"]), str(child.depth), str(policy), str(available_actions),
            str(p), str(ps), str([(c.data["a"], c.solved) for c in node.children]))

        return a, child

    def rollout(self, tree, node: Node, a, novelty_table, random):
        while not self.should_stop(tree):
            node = self.generate_successor(tree, node, a)
            assert node is not None, "Successor function not properly defined?"  # Can't use AbstractTreeActor here
            self.maybe_solve(node, novelty_table, caching=False)

            if node.solved:
                break
            a, child = self.select_action_following_policy(node, random=random)
            assert a is not None and child is None, "Action: %s, child: %s" % (str(a), str(child))
        return node