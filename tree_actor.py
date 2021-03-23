import numpy as np
from tree import Tree
from utils import env_has_wrapper, display_image_cv2
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

    def render(self, tree, size=None, window_name="Render"):
        obs = tree.root.data["obs"]
        img = obs[-1] if type(obs) is list else obs

        if size: img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        display_image_cv2(window_name, img)

    def render_downsampled(self, tree, max_pix_value, size=None, window_name="Render downsampled"):
        if "downsampled_image" in tree.root.data:
            img = tree.root.data["downsampled_image"]
            if size: img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            display_image_cv2(window_name, img/float(max_pix_value))

    def render_tree(self, tree, size=None, window_name="Render tree"):
        get_img = lambda obs: obs[-1] if type(obs) is list else obs
        root_img = get_img(tree.root.data["obs"])
        image = root_img / 255.0
        for child in tree.root.children:
            for node in child.breadth_first():
                image += 0.4 / 255.0 * (get_img(node.data["obs"])-root_img)
        display_image_cv2(window_name, cv2.resize(image, size, interpolation=cv2.INTER_AREA))

    def compute_returns(self, tree, discount_factor, add_value, use_value_all_nodes=False):
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


class AbstractTreeActor:
    def __init__(self, low_level_planner, low_level_tree_actor):
        #TODO: add observe_fns here as well, make tree actors "stackable", as many levels as we wish
        self.low_level_planner = low_level_planner
        self.low_level_tree_actor = low_level_tree_actor
        self.low_level_tree_actor.add_observe_fn(self._low_level_observe_abstract)
        self.low_level_planner.add_stop_fn(lambda tree: self._new_node is not None)
        self.low_level_planner.add_pruning_fn(lambda node: node.entry_point is not None)


    def get_tree_size(self, abstract_tree):
        return abstract_tree._size

    def _new_abstract_node_from_low_level_node(self, abstract_tree, abstract_parent, low_level_node):
        new_abstract_node = abstract_tree.add(abstract_parent,
                                              data={"done": low_level_node.data["done"],
                                                    "high_level_features": low_level_node.data["high_level_features"]})
        new_abstract_node.low_level_tree = Tree(low_level_node.data) # TODO: do not create Tree here, do it from the "low_level_parent" when expanding this node
        new_abstract_node.low_level_tree.root.entry_point = None
        new_abstract_node.low_level_parent = low_level_node
        new_abstract_node.plan_initialized = False
        low_level_node.entry_point = new_abstract_node
        return new_abstract_node

    def _low_level_observe_abstract(self, node):
        if hasattr(self, "_current_tree"):
            self._current_tree._size += 1

        node.entry_point = None
        if not node.is_root() and \
                not node.data["done"] and \
                node.data["high_level_features"] != self._current_node.data["high_level_features"]:
            self._new_node = self._new_abstract_node_from_low_level_node(abstract_tree=self._current_tree,
                                                                         abstract_parent=self._current_node,
                                                                         low_level_node=node)

    def reset(self):
        low_level_tree = self.low_level_tree_actor.reset()
        low_level_tree.root.entry_point = None
        tree = Tree(root_data={"done": False,
                               "high_level_features": low_level_tree.root.data["high_level_features"]})
        tree.root.low_level_tree = low_level_tree
        tree.root.low_level_parent = None
        tree.root.plan_initialized = False
        tree._size = 1
        return tree

    def generate_successor(self, tree, abstract_node, action=None):
        assert action is None, "Action cannot be specified when generating nodes with AbstractTreeActor"

        self._current_tree = tree
        self._current_node = abstract_node # children will be added to self._current_node in observe function
        self._new_node = None # observe function will put here the new node
        if not abstract_node.plan_initialized: # only initialize if a step has been taken
            self.low_level_planner.initialize(tree=abstract_node.low_level_tree)
            abstract_node.plan_initialized = True
        self.low_level_planner.plan(tree=abstract_node.low_level_tree)
        return self._new_node

    def step(self, tree, low_level_action, cache_subtree):
        abstract_node = tree.root

        # Take step in low level tree
        # TODO: if child node is entry point, directly take a step in abstract tree
        # TODO: check if low_level_tree_actor is Env or Abstract and call low level step() if it's the latter
        previous_root_data, new_root = self.low_level_tree_actor.step(abstract_node.low_level_tree, low_level_action, cache_subtree)

        # Remove unreachable abstract node children from abstract tree
        unreachable_abstract_children = [c for c in abstract_node.children if c.low_level_parent not in abstract_node.low_level_tree]
        for c in unreachable_abstract_children:
            tree.remove_subtree(c)

        # Maybe take step in abstract tree
        abstract_child = new_root.entry_point
        if abstract_child is not None:
            assert len(abstract_node.low_level_tree) == 1
            tree.new_root(abstract_child, keep_subtree=cache_subtree)
        else:
            # We took a step, let's reinitialize the low-level plan. It is not necessary to do it if we take a step in
            # the abstract level, since the low-level tree hasn't changed.
            tree.root.plan_initialized = False  # after taking a step, the tree has changed and we need to reinitialize the low-level planner

        tree._size = 1  # count first root
        for node in tree:
            tree._size += self.low_level_tree_actor.get_tree_size(node.low_level_tree)-1  # -1 to discount all roots
        return previous_root_data, tree.root.low_level_tree.root

    def render(self, tree, size=None, window_name="Render"):
        obs = tree.root.low_level_tree.root.data["obs"]
        img = obs[-1] if type(obs) is list else obs

        if size: img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        display_image_cv2(window_name, img)

    def render_downsampled(self, tree, max_pix_value, size=None, window_name="Render downsampled"):
        if "downsampled_image" in tree.root.low_level_tree.root.data:
            img = tree.root.low_level_tree.root.data["downsampled_image"]
            display_image_cv2(window_name, img/float(max_pix_value))

    def _get_image_from_subtree(self, node, background):
        get_img = lambda obs: obs[-1] if type(obs) is list else obs

        image = np.zeros_like(get_img(node.data["obs"]), dtype=np.float32)
        for n in node.breadth_first():
            if n.entry_point is not None:
                image += self._get_image_from_subtree(n.entry_point.low_level_tree.root, background)
            else:
                image += 0.1 / 255.0 * (get_img(n.data["obs"])-background)
        return image

    def render_tree(self, abstract_tree, size=None, window_name="Render tree"):
        get_img = lambda obs: obs[-1] if type(obs) is list else obs

        root = abstract_tree.root.low_level_tree.root
        root_img = get_img(root.data["obs"])
        image = root_img/255.0
        for child in root.children:
            image += self._get_image_from_subtree(child, background=root_img)
        if size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        display_image_cv2(window_name, image)

    def _compute_return_low_level(self, low_level_tree, discount_factor, add_value, use_value_all_nodes):
        for node in low_level_tree.iter_breadth_first_reverse():
            if node.is_leaf():
                R = 0
                if not node.data["done"]:
                    if node.entry_point is not None:
                        R = node.entry_point.data["R"]  # get reward from high-level tree
                    elif add_value:
                        R = node.data["v"]
            else:
                if add_value and use_value_all_nodes:
                    action_returns = [child.data["r"] + discount_factor * child.data["R"] for child in node.children]
                    action_returns += [node.data["v"]]
                    R = np.max(action_returns)
                else:
                    R = np.max([child.data["r"] + discount_factor * child.data["R"] for child in node.children])
            node.data["R"] = R
        return node.data["R"]

    def compute_returns(self, tree, discount_factor, add_value, use_value_all_nodes=False):
        for abstract_node in tree.iter_breadth_first_reverse():
            abstract_node.data["R"] = self._compute_return_low_level(abstract_node.low_level_tree, discount_factor, add_value, use_value_all_nodes)

    def get_counts(self, tree, n_actions):
        # Compute counts of abstract root node's tree
        counts = np.zeros(n_actions)
        for c in tree.root.low_level_tree.root.children:
            counts[c.data["a"]] = c.size()

        # Compute counts for all other abstract nodes
        ac_counts = [sum([self.low_level_tree_actor.get_tree_size(an.low_level_tree) - 1 for an in ac.breadth_first()])
                     for ac in tree.root.children]

        # Add counts of the abstract tree to corresponding low level action
        for ac, cnt in zip(tree.root.children, ac_counts):
            n = ac.low_level_parent
            while not n.is_root():
                a = n.data["a"]
                n = n.parent
            counts[a] += cnt

        return counts