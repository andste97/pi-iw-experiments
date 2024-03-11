import numpy as np
from utils.utils import sample_pmf, reward_in_tree
from utils.utils import softmax
from utils.interactions_counter import InteractionsCounter
from planners.rollout_IW import RolloutIW
import timeit

def get_gridenvs_BASIC_features_fn(env, features_name="features"):
    def gridenvs_BASIC_features(node):
        node.data[features_name] = tuple(enumerate(env.unwrapped.get_char_matrix().flatten()))

    return gridenvs_BASIC_features


def sample_best_action(node, n_actions, discount_factor):
    Q = np.empty(n_actions, dtype=np.float32)
    Q.fill(-np.inf)
    for child in node.children:
        Q[child.data["a"]] = child.data["r"] + discount_factor * child.data["R"]
    return Q


if __name__ == "__main__":
    import gym
    from tree_utils.tree_actor import EnvTreeActor
    import gridenvs.examples  # register GE environments to gym

    # import gridenvs.examples  # gym registration always gets lost during refactoring

    # from pddl2gym.env import PDDLEnv
    # from pddl2gym.simulator import PDDLProblemSimulator
    # from pddl2gym.utils import parse_problem

    # HYPERPARAMETERS
    # env_id can either be a gym environment identifier or a tuple (domain, instance) pddl path files
    env_id = "GE_MazeKeyDoor-v0"
    # env_id = ("../pddl-benchmarks/gripper/domain.pddl", "../pddl-benchmarks/gripper/prob01.pddl")
    interactions_budget = 500
    width = 1  # if we use width 1 here, the planner fails. This is a width 2 problem so this is expected.
    seed = 0

    discount_factor = 0.99
    cache_subtree = True

    frametime = 1  # in milliseconds to display renderings

    nodes_generated = []
    times = []
    rewards = []
    start_time = timeit.default_timer()
    np.random.seed(seed)

    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
    env = gym.make(env_id)

    interactions = InteractionsCounter(budget=interactions_budget)

    # create observe functions
    increase_interactions_fn = lambda node: interactions.increment()
    compute_features_fn = get_gridenvs_BASIC_features_fn(env)

    env_actions = list(range(env.action_space.n))
    applicable_actions_fn = lambda n: env_actions

    actor = EnvTreeActor(env,
                         observe_fns=[
                             compute_features_fn,
                             increase_interactions_fn
                         ],
                         applicable_actions_fn=applicable_actions_fn)

    planner = RolloutIW(generate_successor_fn=actor.generate_successor, width=width,
                        branching_factor=env.action_space.n)
    # planner = IW(generate_successor_fn=actor.generate_successor, width=width, ignore_terminal_nodes=True)
    # planner = BFS(generate_successor_fn=actor.generate_successor)
    # planner = CountbasedRolloutIW(generate_successor_fn=actor.generate_successor, width=width)

    tree = actor.reset()
    episode_done = False
    planner.add_stop_fn(lambda tree: not interactions.within_budget() or reward_in_tree(tree))
    steps_cnt = 0

    while not episode_done:
        interactions.reset_budget()
        planner.initialize(tree=tree)
        planner.plan(tree=tree)

        actor.compute_returns(tree, discount_factor=discount_factor, add_value=False)
        #actor.render_tree(tree, size=(800, 800), frametime=frametime)

        Q = sample_best_action(node=tree.root,
                               n_actions=env.action_space.n,
                               discount_factor=discount_factor)

        policy_output = softmax(Q, temp=0)
        action = sample_pmf(policy_output)

        prev_root_data, current_root = actor.step(tree, action, cache_subtree=cache_subtree)

        current_root_data = current_root.data
        episode_done = current_root_data["done"]
        steps_cnt += 1

        print("\n".join([" ".join(row) for row in
                         env.unwrapped.world.get_char_matrix(
                             env.unwrapped.get_gridstate(tree.root.data["s"]["state"]))
                         ]),
              "Action: ", current_root_data["a"], "Reward: ", current_root_data["r"],
              "Simulator steps:", interactions.value, "Planning steps:", steps_cnt, "\n")

    print("It took %i steps but the problem can be solved in 13." % steps_cnt)
