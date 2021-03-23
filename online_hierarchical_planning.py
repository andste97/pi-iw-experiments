import numpy as np
import cv2
from utils import sample_pmf


if __name__ == "__main__":
    import envs
    from rollout_IW import RolloutIW
    from IW import IW
    from bfs import BFS
    from countbased_rollout_iw import CountbasedRolloutIW
    from pi_HIW import compute_node_Q, get_downsampled_features_fn, make_env
    from tree_actor import EnvTreeActor, AbstractTreeActor
    from planning_step import get_gridenvs_BASIC_features_fn
    from utils import Stats, softmax, InteractionsCounter
    import gridenvs.examples #load simple envs


    # HYPERPARAMETERS
    seed = 1
    env_id = "GE_MKDL-v0"
    max_steps = 1000
    interactions_budget = 30
    discount_factor = 0.99
    cache_subtree = True

    low_level_width = 1
    high_level_width = 1
    downsampling_shape = (4, 4)
    downsampling_pix_values = 256

    display_time = 100  # in milliseconds

    # Set random seed
    np.random.seed(seed)

    env, _ = make_env(env_id, max_episode_steps=max_steps, add_downsampling=True,
                      downsampling_tiles_w=downsampling_shape[0],
                      downsampling_tiles_h=downsampling_shape[1],
                      downsampling_pix_values=downsampling_pix_values,
                      atari_frameskip=15)
    env_actions = list(range(env.action_space.n))
    applicable_actions_fn = lambda n: env_actions

    interactions = InteractionsCounter(budget=interactions_budget)

    observe_fns = list()
    observe_fns.append(lambda node: interactions.increment())
    gridenvs_BASIC_features = get_gridenvs_BASIC_features_fn(env, features_name="low_level_features")
    observe_fns.append(gridenvs_BASIC_features)
    high_level_feats_fn = get_downsampled_features_fn(env, features_name="high_level_features")
    observe_fns.append(high_level_feats_fn)

    low_level_tree_actor = EnvTreeActor(env=env,
                                        applicable_actions_fn=applicable_actions_fn,
                                        observe_fns=observe_fns)

    # low_level_planner = BFS(generate_successor_fn=low_level_tree_actor.generate_successor, features_name="low_level_features")
    low_level_planner = RolloutIW(generate_successor_fn=low_level_tree_actor.generate_successor, branching_factor=env.action_space.n, width=low_level_width, features_name="low_level_features")
    # low_level_planner = IW(generate_successor_fn=low_level_tree_actor.generate_successor, width=low_level_width, features_name="low_level_features")
    # low_level_planner = CountbasedRolloutIW(generate_successor_fn=low_level_tree_actor.generate_successor, width=low_level_width, features_name="low_level_features")

    low_level_planner.add_stop_fn(lambda tree: not interactions.within_budget())

    abstract_tree_actor = AbstractTreeActor(low_level_planner=low_level_planner,
                                            low_level_tree_actor=low_level_tree_actor)

    # high_level_planner = BFS(generate_successor_fn=abstract_tree_actor.generate_successor, features_name="high_level_features")
    # high_level_planner = IW(generate_successor_fn=abstract_tree_actor.generate_successor, width=high_level_width, features_name="high_level_features")
    high_level_planner = CountbasedRolloutIW(generate_successor_fn=abstract_tree_actor.generate_successor, width=high_level_width, features_name="high_level_features")

    high_level_planner.add_stop_fn(lambda tree: not interactions.within_budget())

    abstract_tree = abstract_tree_actor.reset()
    episode_done = False
    stats = Stats()
    abstract_tree_actor.render_tree(abstract_tree, size=None)
    while not episode_done:
        interactions.reset_budget()
        high_level_planner.initialize(tree=abstract_tree)
        high_level_planner.plan(tree=abstract_tree)

        abstract_tree_actor.compute_returns(abstract_tree, discount_factor=discount_factor, add_value=False)
        Q = compute_node_Q(node=abstract_tree.root.low_level_tree.root,
                           n_actions=env.action_space.n,
                           discount_factor=discount_factor,
                           add_value=False)
        low_level_policy = softmax(Q, temp=0)
        a = sample_pmf(low_level_policy)
        abstract_tree_nodes = len(abstract_tree)

        abstract_tree_actor.render_tree(abstract_tree, size=None)
        prev_root_data, current_root = abstract_tree_actor.step(abstract_tree, a, cache_subtree=cache_subtree)

        episode_done = current_root.data["done"]

        stats.increment("planning_steps", step=interactions.value)
        stats.add({"action": current_root.data["a"],
                   "reward": current_root.data["r"],
                   "abstract_tree_nodes": abstract_tree_nodes,},
                   step=interactions.value)

        stats.report()
        cv2.waitKey(display_time)  # wait time in ms

    print("\nIt took %i steps." % stats.get_last("planning_steps"))

