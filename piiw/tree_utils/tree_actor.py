import numpy as np
from tree_utils.tree import Tree
from utils.utils import env_has_wrapper, display_image_cv2
import cv2


class EnvTreeActor:
    """
    Interacts with an environment while adding nodes to a tree.
    """

    def __init__(self, env, observe_fns=None, applicable_actions_fn=None):
        self.env = env
        self.applicable_actions_fn = applicable_actions_fn
        if observe_fns is None:
            observe_fns = list()
        else:
            assert isinstance(observe_fns, (list, tuple))
        self.observe_fns = observe_fns
        self.simulator_node = None

        # gym usually puts a TimeLimit wrapper around an env when creating it with gym.make(). In our case this is not
        # desired since we will most probably reach the step limit (the step count will not reset when restoring the
        # internal state).
        import gym.wrappers
        assert not env_has_wrapper(self.env, gym.wrappers.TimeLimit)

    def get_tree_size(self, tree):
        return len(tree)

    def add_observe_fn(self, fn):
        self.observe_fns.append(fn)

    def generate_successor(self, tree, node, action=None):
        assert node in tree
        assert not node.data["done"], "Trying to generate a successor from a terminal node (the episode is over)."
        if hasattr(node, "entry_point"): # Maybe we are using AbstractTreeActor, check if we are doing it right
            assert node.entry_point is None, "Trying to generate a successor from an entry point."

        if action is None: # compatibility with planners that do not specify action
            assert self.applicable_actions_fn is not None, "Either specify an action or initialize EnvTreeActor with " \
                                                           "all the possible actions. In this last setting, an action that has not previously been selected for " \
                                                           "the given node will be chosen at random at each generate_successor call."
            non_expanded_actions = list(set(self.applicable_actions_fn(node)) - set([child.data["a"] for child in node.children]))
            if len(non_expanded_actions) == 0:
                return None
            action = np.random.choice(non_expanded_actions)
            # action = non_expanded_actions[0]

        if self.simulator_node is not node:
            self.env.restore_state(node.data["s"])

        # Perform step
        next_obs, r, end_of_episode, info = self.env.step(action)
        child_node_data = {"a": action, "r": r, "done": end_of_episode, "obs": next_obs}
        child_node_data.update(info) # add extra info e.g. atari lives
        child = tree.add(node, child_node_data)
        self._observe(child) # set simulator node to child

        return child

    def reset(self):
        obs = self.env.reset()
        tree = Tree({"obs": obs, "done": False, "r": None})
        self._observe(tree.root)
        return tree

    def _observe(self, node):
        node.data["s"] = self.env.clone_state()
        for fn in self.observe_fns:
            fn(node)
        self.simulator_node = node
        # self.counter["interactions"] += 1

    def step(self, tree, a, cache_subtree):
        """Set the next node as tree root according to the taken action a

        :returns: Tuple of old root data and next root node."""

        assert not tree.root.data["done"], "Trying to take a step, but either the episode is over or hasn't " \
                                                "started yet. Please use reset()."
        next_node = self._get_next_node(tree, a)
        root_data = tree.root.data

        # "take a step" (actually remove other branches and make selected child root)
        tree.new_root(next_node, keep_subtree=cache_subtree)
        return root_data, next_node

    def _get_next_node(self, tree, a):
        assert not tree.root.is_leaf()

        next_node = None
        for child in tree.root.children:
            if a == child.data["a"]:
                next_node = child
        assert next_node is not None, "Selected action not in tree. Something wrong with the lookahead policy?"

        return next_node

    def render(self, tree, size=(800,800), window_name="Render", frametime=1000):
        obs = tree.root.data["obs"]
        img = obs[-1] if type(obs) is list else obs

        # refactor img to put channels last for cv2
        img = np.moveaxis(img, 0, -1)
        if size: img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # refactor image again for other used libraries
        img = np.moveaxis(img, -1, 0)
        return img

    def render_downsampled(self, tree, max_pix_value, size=None, window_name="Render downsampled"):
        if "downsampled_image" in tree.root.data:
            img = tree.root.data["downsampled_image"]
            if size: img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            display_image_cv2(window_name, img/float(max_pix_value))

    def render_tree(self, tree, size=(800,800), window_name="Render tree", frametime=10):
        """Renders and displays the entire planning tree as a succession of images in windoes.
        Should only be used with the gridenvs environment."""
        get_img = lambda obs: obs[-1] if type(obs) is list else obs
        root_img = get_img(tree.root.data["obs"])
        image = root_img / 255.0
        for child in tree.root.children:
            for node in child.breadth_first():
                image += 0.4 / 255.0 * (get_img(node.data["obs"])-root_img)
                display_image_cv2(window_name, cv2.resize(image, size, interpolation=cv2.INTER_AREA), block_ms=frametime)

    def compute_returns(self, tree, discount_factor, add_value, use_value_all_nodes=False):
        """Computes rewards for entire tree, starting with last node in tree.

        If add_value is true, the value from predictions is added to """

        # R is the discounted accumulated max reward of the children
        # with distance discounted by discount_factor
        # each level deeper gets discounted once by multiplying with discount_factor.
        for node in tree.iter_breadth_first_reverse():
            if node.is_leaf():
                R = 0
                if add_value and not node.data["done"]:
                    R = node.data["v"]
            else:
                if add_value and use_value_all_nodes:
                    action_returns = [child.data["r"] + discount_factor * child.data["R"] for child in node.children]
                    action_returns += [node.data["v"]]
                    R = np.max(action_returns)
                else:
                    R = np.max([child.data["r"] + discount_factor * child.data["R"] for child in node.children])
            node.data["R"] = R

    def get_counts(self, tree, n_actions):
        counts = np.zeros(n_actions)
        for c in tree.root.children:
            counts[c.data["a"]] = c.size()
        return counts