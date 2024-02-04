from itertools import combinations


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
