from collections import deque

import numpy as np

from planners.planner import Planner
from utils.utils import sample_pmf


class BFSBestFirstPlanner(Planner):
    def __init__(self, generate_successor_fn, branching_factor, policy_fn=None, min_cum_prob=0):
        """
        :param branching_factor: Number of possible actions
        :param width: Tuples of features of this length will be considered for novelty check
        :param ignore_cached_nodes: If set to True, nodes already existing in the tree will be ignored and their features will not be added to the novelty table
        :param ignore_terminal_nodes: If set to True, terminal nodes (episode done/game over) will be ignored and their features will not be added to the novelty table
        :param min_cum_prob: After discarding all actions that have been solved, if the sum of probabilities for the remaining actions is less than min_cum_prob, the current node will be pruned (set to solved).
        :param policy_fn: Given a node and the number of possible actions, it returns a policy (probability distribution) that will be used for traversing the tree and for generating new nodes.
        """
        super(BFSBestFirstPlanner, self).__init__(generate_successor_fn=generate_successor_fn)
        self.branching_factor = branching_factor

        if policy_fn is None:
            policy_fn = lambda n: np.full(self.branching_factor, 1 / self.branching_factor)

        self.policy_fn = policy_fn
        self.min_cum_prob = min_cum_prob  # Prune when the cumulative probability of the remaining (not solved) actions is lower than this threshold

    def plan(self, tree):
        """
        :param tree: Tree to begin expanding nodes. It can contain just the root node (for offline planning or online
        planning without caching nodes), or an many (cached) nodes.
        """

        # Add nodes of the tree to the queue
        queue = deque()
        visited_nodes = set()
        for node in tree.iter_breadth_first():
            if not hasattr(node, "expanded"):
                # Assume the node has not been expanded. If it actually is, generate_successor() will return False,
                # it will be marked as expanded, and the search will continue normally
                node.expanded = False
            if not node.expanded and not node.data["done"] and not self.should_prune(node):
                queue.append(node)

        # Expand nodes
        while len(queue) != 0 and not self.should_stop(tree):
            node = queue.popleft()
            while not self.should_stop(tree):
                a, _ = self.select_action_following_policy(node)
                child = self.generate_successor(tree, node, a)
                if child is None:
                    node.expanded = True
                    break
                else:
                    child.expanded = False
                    if not child.data["done"] and not self.should_prune(child):
                        queue.append(child)

    def select_action_following_policy(self, node):
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
        policy = self.policy_fn(node)

        if node.is_leaf():
            # return action to expand
            assert not node.data["done"], "Done: %s.  Depth: %s" % (
            str(node.data["done"]), str(node.depth))
            return sample_pmf(policy), None

        node_children = [None] * self.branching_factor
        available_actions = (policy > 0)
        for child in node.children:
            node_children[child.data["a"]] = child
            if child.data["done"]:
                available_actions[child.data["a"]] = False

        # Take out actions that have been solved
        p = (policy * available_actions)
        ps = p.sum()

        # Select action not solved
        p = p / ps
        assert all((p > 0) == available_actions), "p: %s;  avail_a: %s;   ps:%s" % (
        str(p), str(available_actions), str(ps))
        a = sample_pmf(p)

        child = node_children[a]
        if child:
            assert  not child.data[
                "done"], "a: %i,  Done: %s.  Depth: %s.  policy: %s.  avail_actions: %s.  p: %s.  ps: %s.  children: %s." % (
            a, str(child.data["done"]), str(child.depth), str(policy), str(available_actions),
            str(p), str(ps), str([(c.data["a"]) for c in node.children]))

        return a, child