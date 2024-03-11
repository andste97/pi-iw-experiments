import numpy as np

def reward_in_tree(tree):
    """Check if any nodes within this tree contain a reward

    :return True if any nodes within this tree contain a reward, False otherwise
    """
    iterator = iter(tree)
    next(iterator) # discard root
    for node in iterator:
        if node.data["r"] > 0:
            return True
    return False

# Define how we will extract features
def get_gridenvs_BASIC_features_fn(env, features_name="features"):
    def gridenvs_BASIC_features(node):
        node.data[features_name] = tuple(enumerate(env.unwrapped.get_char_matrix().flatten()))
    return gridenvs_BASIC_features

def get_state_atoms_fn(features_name="features"):
    def state_atoms(node):
        node.data[features_name] = node.data["obs"]
    return state_atoms

def discover_atoms(node, features_name):
    novel_atoms_branch = set()
    if not node.is_root() and node.is_leaf() and not node.data["done"] and node.pruned and node.depth > 1:
        atoms_leaf_and_parent = set(node.data[features_name]).intersection(set(node.parent.data[features_name]))  # get common atoms with parent
        if len(atoms_leaf_and_parent) < len(node.data[features_name]):  # check it's not the same state as the parent
            atoms_branch = set.union(*(set(n.data[features_name]) for n in node.parent.ascendants()))
            novel_atoms_branch = atoms_leaf_and_parent - atoms_branch
    return novel_atoms_branch

def discover_atoms_tree(tree, features_name):
    novel_atoms = set()
    for node in tree:
        novel_atoms_branch = discover_atoms(node, features_name)
        if len(novel_atoms_branch) > 0:
            novel_atoms.update(novel_atoms_branch)
    return novel_atoms


if __name__ == "__main__":
    import gym
    from tree_utils.tree_actor import EnvTreeActor
    from planners.rollout_IW import RolloutIW
    from utils import InteractionsCounter
    import timeit
    import gridenvs.examples  # register GE environments to gym
    # import gridenvs.examples  # gym registration always gets lost during refactoring

    #from pddl2gym.env import PDDLEnv
    #from pddl2gym.simulator import PDDLProblemSimulator
    #from pddl2gym.utils import parse_problem


    # HYPERPARAMETERS
    # env_id can either be a gym environment identifier or a tuple (domain, instance) pddl path files
    env_id = "GE_MazeKeyDoor-v0"
    # env_id = ("../pddl-benchmarks/gripper/domain.pddl", "../pddl-benchmarks/gripper/prob01.pddl")
    max_tree_nodes = 10000
    width = 1 # if we use width 1 here, the planner fails. This is a width 2 problem so this is expected.
    seed = 0

    nodes_generated = []
    times = []
    rewards = []
    start_time = timeit.default_timer()
    np.random.seed(seed)

    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
    env = gym.make(env_id)
    env_actions = list(range(env.action_space.n))
    applicable_actions_fn = lambda n: env_actions
    compute_features_fn = get_gridenvs_BASIC_features_fn(env)
    #else:
    #    domain_path, instance_path = env_id
    #    env = PDDLEnv(PDDLProblemSimulator(parse_problem(domain_path, instance_path)))
    #    compute_features_fn = get_state_atoms_fn()
    #    applicable_actions_fn = lambda n: env.simulator.get_applicable_str_actions(n.data["obs"])

    interactions = InteractionsCounter(budget=max_tree_nodes)
    actor = EnvTreeActor(env,
                         observe_fns=[
                            compute_features_fn,
                            lambda _: interactions.increment()
                         ],
                         applicable_actions_fn=applicable_actions_fn)

    planner = RolloutIW(generate_successor_fn=actor.generate_successor, width=width, branching_factor=env.action_space.n)
    # planner = IW(generate_successor_fn=actor.generate_successor, width=width, ignore_terminal_nodes=True)
    # planner = BFS(generate_successor_fn=actor.generate_successor)
    # planner = CountbasedRolloutIW(generate_successor_fn=actor.generate_successor, width=width)

    planner.add_stop_fn(lambda tree: not interactions.within_budget() or reward_in_tree(tree))

    tree = actor.reset()
    planner.initialize(tree=tree)
    planner.plan(tree=tree)
    nodes_generated = len(tree)
    time = timeit.default_timer() - start_time
    print("Planner:", planner.__class__.__name__)
    print("Width:", width)
    print("Env:", env_id)
    print("Nodes generated:", nodes_generated)
    print("Time:", time)
    print("interactions: ", interactions.value)

    actor.render_tree(tree, size=(800, 800))
    # print(tree.root.str_tree())

    if reward_in_tree(tree):
        print("\n================\n|    SOLVED    |\n================")
    else:
        print("\n================\n|      :(      |\n================\n")

    if hasattr(tree.root, "pruned"):
        print("Features discovered:")
        novel_features_H = discover_atoms_tree(tree, features_name="features")
        print(novel_features_H)