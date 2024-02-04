class InteractionsCounter:
    """Counts numbers of interactions with something."""
    def __init__(self, budget):
        self.budget = budget
        self.value = 0
        self.start_value = 0

    def increment(self):
        self.value += 1

    def within_budget(self):
        return (self.value - self.start_value) < self.budget

    def reset_budget(self):
        self.start_value = self.value
