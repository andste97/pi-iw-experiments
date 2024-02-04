class Novelty1Table():
    def __init__(self):
        self.table = set()

    def check(self, atoms):
        return any(atom for atom in atoms if atom not in self.table)

    def check_and_update(self, atoms):
        l = len(self.table)
        self.table.update(atoms)
        return len(self.table) != l
