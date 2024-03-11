from planners.width_planner import WidthPlanner
from novelty_tables.binary_novelty_table import BinaryNoveltyTable
from novelty_tables.depth_novelty_n_table import DepthNoveltyNTable
from novelty_tables.depth_features_novelty_1_table import DepthFeaturesNovelty1Table
from novelty_tables.depth_atoms_novelty_1_table import DepthAtomsNovelty1Table


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
