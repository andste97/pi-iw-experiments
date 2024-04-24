from planners.planner import Planner
from tree_utils.node import Node


class WidthPlanner(Planner):
    def __init__(self, generate_successor_fn, width, ignore_cached_nodes, ignore_terminal_nodes, features_name):
        super(WidthPlanner, self).__init__(generate_successor_fn)
        self.width = width
        self.ignore_cached_nodes = ignore_cached_nodes
        self.ignore_terminal_nodes = ignore_terminal_nodes
        self.features_name = features_name

    def check_update_novelty(self, node: Node, novelty_table, caching):
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