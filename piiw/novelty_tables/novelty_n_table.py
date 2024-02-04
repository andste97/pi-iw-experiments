class NoveltyNTable:
    def __init__(self):
        self.visited = set()

    def check(self, atoms):
        if atoms in self.visited:
            return False
        return True

    def check_and_update(self, atoms):
        """
        Evaluates the novelty of a state up to the pre-set max-width.
        """
        if atoms in self.visited:
            return False
        self.visited.add(atoms)
        return True
