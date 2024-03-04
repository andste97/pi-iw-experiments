from planners.width_planner import WidthPlanner
from novelty_tables.binary_novelty_table import BinaryNoveltyTable
from novelty_tables.novelty_n_table import NoveltyNTable
from novelty_tables.novelty_1_table import Novelty1Table


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
