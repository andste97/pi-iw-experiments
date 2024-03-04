from planners.rollout_width_planner import RolloutWidthPlanner


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
