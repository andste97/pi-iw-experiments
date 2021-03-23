class Planner:
    def __init__(self, generate_successor_fn):
        """
        :param generate_successor_fn: function to generate a successor node (e.g. interact with the simulator to
        generate a new state).
        """
        self.generate_successor = generate_successor_fn
        self._stop_fns = []
        self._pruning_fns = []

    def add_stop_fn(self, stop_fn):
        """
        :param stop_fn: Evaluates to True when the planning should stop (e.g. nodes/time budget exhausted)
        """
        self._stop_fns.append(stop_fn)

    def should_stop(self, *args):
        return any(f(*args) for f in self._stop_fns)

    def add_pruning_fn(self, pruning_fn):
        self._pruning_fns.append(pruning_fn)

    def should_prune(self, *args):
        return any(f(*args) for f in self._pruning_fns)

    def initialize(self, tree):
        pass

    def plan(self, tree):
        raise NotImplementedError()

