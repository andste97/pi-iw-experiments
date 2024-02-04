from collections import deque
from planner import Planner

class BFS(Planner):
    def __init__(self, generate_successor_fn, features_name="features", expand_all_nodes=False):
        """
        :param generate_successor_fn: function to generate a successor node (e.g. interact with the simulator to
        generate a new state).
        :param features_name:
        """
        super(BFS, self).__init__(generate_successor_fn)
        self.features_name = features_name
        self.expand_all_nodes = expand_all_nodes

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
            if node.data[self.features_name] not in visited_nodes or self.expand_all_nodes:
                while not self.should_stop(tree):
                    child = self.generate_successor(tree, node)
                    if child is None:
                        node.expanded=True
                        break
                    else:
                        child.expanded=False
                        if not child.data["done"] and not self.should_prune(child):
                            queue.append(child)
                visited_nodes.add(node.data[self.features_name])