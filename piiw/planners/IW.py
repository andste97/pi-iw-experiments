from collections import deque
from planners.BFSWidthPlanner import BFSWidthPlanner


class IW(BFSWidthPlanner):
    def __init__(self, generate_successor_fn, width=1, ignore_cached_nodes=False, ignore_terminal_nodes=False, features_name="features"):
        """
        :param generate_successor_fn: function to generate a successor node (e.g. interact with the simulator to
        generate a new state).
        """
        super(IW, self).__init__(generate_successor_fn=generate_successor_fn,
                                 width=width,
                                 ignore_cached_nodes=ignore_cached_nodes,
                                 ignore_terminal_nodes=ignore_terminal_nodes,
                                 features_name=features_name)

    def initialize(self, tree):
        novelty_table = self.create_novelty_table()

        queue = deque()
        for node in tree.iter_insertion_order():
            assert self.features_name in node.data.keys(), "IW planners require the state to be factored into features"
            if self.check_update_novelty(node, novelty_table, caching=True):
                queue.append(node)

        tree.root.queue = queue
        tree.root.novelty_table = novelty_table

    def plan(self, tree):
        """
        :param tree: Tree to begin expanding nodes. It can contain just the root node (for offline planning or online
        planning without caching nodes), or an many (cached) nodes.
        """

        # Initialize queue and novelty table, by adding the existing nodes in the tree
        assert hasattr(tree.root, "novelty_table"), "Planner not initialized."
        queue, novelty_table = tree.root.queue, tree.root.novelty_table

        # Expand nodes
        while not self.should_stop(tree) and len(queue) != 0:
            node = queue[0]
            child = self.generate_successor(tree, node)
            if child is None:
                queue.popleft()
            else:
                if self.check_update_novelty(child, novelty_table, caching=False):
                    queue.append(child)