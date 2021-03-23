import os
import numpy as np
import random
import tensorflow as tf
import logging
import time
import gym, gym.wrappers

from utils import sample_pmf, Stats, remove_env_wrapper, env_has_wrapper, softmax, ParamsDef, InteractionsCounter
from planning_step import get_gridenvs_BASIC_features_fn
from training import get_loss_fn, get_train_fn, Mnih2013
from experience_replay import ExperienceReplay

from rollout_IW import RolloutIW
from countbased_rollout_iw import CountbasedRolloutIW
from bfs import BFS
from IW import IW

from tree_actor import EnvTreeActor, AbstractTreeActor
from atari_wrappers import is_atari_env, wrap_atari_env, Downsampling
from datetime import datetime
import psutil


job_date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
logger = logging.getLogger(__name__)


# HYPERPARAMETERS
class Params(ParamsDef):
    directory = ParamsDef.NoneDef(str)
    job_id = job_date
    render = False
    render_fps = ParamsDef.NoneDef(float)

    env = "GE_MKDL-v0"
    atari_frameskip = 15

    seed = 789
    hierarchical = True
    low_level_planner = "RolloutIW"  # RolloutIW, BFS, IW
    high_level_planner = "CountbasedRolloutIW"  # BFS, IW
    low_level_width = 1
    high_level_width = 1  # ParamsDef.NoneDef(int)
    compute_value = True
    use_value = True
    features = "dynamic"  # "BASIC"
    use_features_nt = False

    downsampling_tiles_w = 2
    downsampling_tiles_h = 2
    downsampling_pix_values = 256

    max_interactions = 1000000
    interactions_budget = 30
    max_episode_steps = 200  # ParamsDef.NoneDef(int)

    target_policy_temp = 0.01
    tree_policy_temp = 0.01
    eval_temp = 0.01

    eval_episodes = 1
    eval_every_interactions = 100000
    save_every_interactions = 100000

    max_tree_size = 5000  # np.inf

    discount_factor = 0.99
    cache_subtree = True
    batch_size = 32
    learning_rate = 0.0007
    replay_capacity = 10000
    replay_min_transitions = batch_size

    regularization_factor = 0.001
    rmsprop_decay = 0.99
    rmsprop_epsilon = 0.1
    value_factor = 1.0
    ignore_cached_nodes = False
    use_graph = True

    guide_plan_network_policy = True
    learn = True

    use_tensorboard = True
    debug = False

    RIW_ensure_same_init = False
    countbasedRIW_temp = 0.005

    tree_policy_counts_temp = 1.0  #  ParamsDef.NoneDef(float)

    max_grad_norm = 50

    use_value_classification = True
    value_classification_supports = 301
    value_classification_max = 300
    value_classification_min = -300

    use_value_at_init = False
    use_value_all_nodes = True

    save_network = False  #True
    model_dense_units = 256
    use_batch_normalization = True
    use_batch_renorm = True


def make_env(env_id, max_episode_steps, add_downsampling, downsampling_tiles_w, downsampling_tiles_h, downsampling_pix_values, atari_frameskip):
    # Create the gym environment. When creating it with gym.make(), gym usually puts a TimeLimit wrapper around an env.
    # We'll take this out since we will most likely reach the step limit (when restoring the internal state of the
    # emulator the step count of the wrapper will not reset)
    env = gym.make(env_id)
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
        logger.info("TimeLimit environment wrapper removed.")

    from gridenvs.env import GridEnv
    if isinstance(env, GridEnv):
        env.max_moves = max_episode_steps
        if add_downsampling:
            env = Downsampling(env,
                               downsampling_tiles_w=downsampling_tiles_w,
                               downsampling_tiles_h=downsampling_tiles_h,
                               downsampling_pixel_values=downsampling_pix_values)

    # If the environment is an Atari game, the observations will be the last four frames stacked in a 4-channel image
    if is_atari_env(env):
        env = wrap_atari_env(env, frameskip=atari_frameskip, max_steps=max_episode_steps,
                             add_downsampling=add_downsampling, downsampling_tiles_w=downsampling_tiles_w,
                             downsampling_tiles_h=downsampling_tiles_h,
                             downsampling_pix_values=downsampling_pix_values)
        logger.info("Atari environment modified: observation is now a 4-channel image of the last four non-skipped frames in grayscale. Frameskip set to %i." % atari_frameskip)
        preproc_obs_fn = lambda obs_batch: np.moveaxis(obs_batch, 1, -1)  # move channels to the last dimension
    else:
        preproc_obs_fn = lambda x: np.asarray(x)

    return env, preproc_obs_fn

def reward_in_tree(tree):
    if hasattr(tree.root, "low_level_tree"):
        for abstract_node in tree:
            if reward_in_tree(abstract_node.low_level_tree):
                return True
    else:
        iterator = tree.iter_insertion_order()
        next(iterator)  # discard root
        for node in iterator:
            if node.data["r"] > 0:
                return True
    return False


def get_downsampled_features_fn(env, features_name="high_level_features"):
    def downsampled(node):
        node.data["downsampled_image"] = env.downsampled_image #[:-2]
        node.data[features_name] = tuple(enumerate(node.data["downsampled_image"].flatten()))
    return downsampled

def get_observe_nn_fn(model, preproc_obs_fn, get_features, args, value_logits_to_scalars=None):
    def _observe_nn(node):
        x = tf.constant(preproc_obs_fn([node.data["obs"]]).astype(np.float32))
        res = model(x, training=False)
        node.data["probs"] = tf.nn.softmax(res["policy_logits"]).numpy().ravel()
        if "value_logits" in res.keys():
            if value_logits_to_scalars is None:
                node.data["v"] = res["value_logits"].numpy().squeeze()
            else:
                node.data["v"] = value_logits_to_scalars(res["value_logits"].numpy()).squeeze()

        if get_features:
            if args.use_features_nt:
                node.data["low_level_features"] = res["features"].numpy().ravel().astype(np.bool).astype(int)
            else:
                node.data["low_level_features"] = tuple(enumerate(res["features"].numpy().ravel().astype(np.bool)))
    return _observe_nn

def compute_node_Q(node, n_actions, discount_factor, add_value=False):
    Q = np.empty(n_actions, dtype=np.float32)
    if add_value:
        Q.fill(node.data["v"])
    else:
        Q.fill(-np.inf)
    for child in node.children:
        Q[child.data["a"]] = child.data["r"] + discount_factor*child.data["R"]
    return Q

def compute_trajectory_returns(rewards, discount_factor):
    R = 0
    returns = []
    for i in range(len(rewards) - 1, -1, -1):
        R = rewards[i] + discount_factor * R
        returns.append(R)
    return list(reversed(returns))

def process_trajectory(trajectory, add_returns, discount_factor):
    res = {}
    res["observations"] = [node_data["obs"] for node_data in trajectory[:-1]]
    res["target_policy"] = [node_data["target_policy"] for node_data in trajectory[:-1]]

    res["rewards"] = [node_data["r"] for node_data in trajectory[1:]]  # Note the 1: instead of :-1
    if add_returns:
        res["returns"] = compute_trajectory_returns(res["rewards"], discount_factor)  # Backpropagate rewards
    return res

def get_episode_fn(actor, planner, train_fn, dataset, add_returns, stats, memory_usage_fn, preproc_obs_fn, eval_fn,
                   n_actions, value_scalars_to_distrs, value_logits_to_scalars, args):
    interactions.last_eval_interactions = interactions.value

    def run_episode(train, use_value_for_tree_policy):
        reset_counts_fn = getattr(planner, "reset_counts", None)
        if callable(reset_counts_fn):
            reset_counts_fn()

        episode_done = False
        tree = actor.reset()
        if args.hierarchical: # TODO:
            trajectory = [tree.root.low_level_tree.root.data]
        else:
            trajectory = [tree.root.data]
        solved_in_one_planning_step = False
        r_found = False
        while not episode_done:
            time_start_step = time.time()
            interactions_before_step = interactions.value

            # Planning step
            nodes_before_plan = actor.get_tree_size(tree)
            time_start_plan = time.time()
            interactions.reset_budget()
            planner.initialize(tree=tree)
            planner.plan(tree=tree)
            time_plan = time.time() - time_start_plan
            nodes_after_plan = actor.get_tree_size(tree)

            # if len(trajectory) == 1 and reward_in_tree(tree):
            #     solved_in_one_planning_step = True

            if args.hierarchical:
                stats.add({"n_abstract_states_so_far": len(planner.visits.keys()),
                           "n_abstract_states_in_tree": len(
                               set(n.data["high_level_features"] for n in tree)),
                           "n_abstract_nodes_in_tree": len(tree),
                           "avg_nodes_per_abstract_node": nodes_after_plan / len(tree)},
                          step=interactions.value)

            # Execute action (choose one node as the new root from depth 1)
            time_start_execute_action = time.time()

            if args.debug:
                r_in_tree = reward_in_tree(tree)
                r_found = r_found or r_in_tree


            actor.compute_returns(tree, discount_factor=args.discount_factor, add_value=False)
            if args.hierarchical:
                root_node = tree.root.low_level_tree.root
            else:
                root_node = tree.root
            action_returns = compute_node_Q(node=root_node,
                                        n_actions=n_actions,
                                        discount_factor=args.discount_factor,
                                        add_value=False)
            if args.compute_value:
                actor.compute_returns(tree, discount_factor=args.discount_factor, add_value=True, use_value_all_nodes=args.use_value_all_nodes)
                Q = compute_node_Q(node=root_node,
                                   n_actions=n_actions,
                                   discount_factor=args.discount_factor,
                                   add_value=args.use_value_all_nodes)

            # TARGET
            if use_value_for_tree_policy:
                target_policy = softmax(Q, temp=args.target_policy_temp)
            else:
                target_policy = softmax(action_returns, temp=args.target_policy_temp)


            # EXECUTION POLICY
            if use_value_for_tree_policy:
                Q_aux = compute_node_Q(node=root_node,
                                       n_actions=n_actions,
                                       discount_factor=args.discount_factor,
                                       add_value=False)
                tree_policy = softmax(Q_aux, temp=args.tree_policy_temp)
            else:
                tree_policy = softmax(action_returns, temp=args.tree_policy_temp)

            if args.tree_policy_counts_temp is not None:
                counts = actor.get_counts(tree, n_actions)
                counts_policy = softmax(counts, temp=args.tree_policy_counts_temp)
                p = tree_policy * counts_policy
                sum_p = sum(p)
                if sum_p != 0:
                    tree_policy = p / sum_p

            a = sample_pmf(tree_policy)

            if args.render:
                actor.render_tree(tree, size=(512, 512), window_name="Tree before step")

            prev_root_data, current_root = actor.step(tree, a, cache_subtree=args.cache_subtree)

            prev_root_data["target_policy"] = target_policy
            nodes_after_execution = actor.get_tree_size(tree)

            time_execute_action = time.time() - time_start_execute_action
            if args.debug:
                actions_explored = sum(counts > 0)

            if args.render:
                actor.render(tree, size=(512, 512))
                actor.render_tree(tree, size=(512, 512), window_name="Tree after step")
                if args.hierarchical:
                    actor.render_downsampled(tree, max_pix_value=args.downsampling_pix_values, size=(512, 512))

                if args.render_fps is not None:
                    time.sleep(1/args.render_fps)

            trajectory.append(current_root.data)

            episode_done = current_root.data["done"]

            # Learning step
            time_start_learn = time.time()
            if train and len(dataset) > args.batch_size and len(dataset) > args.replay_min_transitions:
                _, batch = dataset.sample(size=args.batch_size)

                if train:
                    input_dict = {"observations": tf.constant(preproc_obs_fn(batch["observations"]), dtype=tf.float32),
                                  "target_policy": tf.constant(batch["target_policy"], dtype=tf.float32)}
                    if args.compute_value:
                        if value_scalars_to_distrs is not None:
                            input_dict["returns"] = tf.constant(value_scalars_to_distrs(batch["returns"]), dtype=tf.float32)
                        else:
                            input_dict["returns"] = tf.constant(batch["returns"], dtype=tf.float32)


                    loss, train_output = train_fn(input_dict)
                    stats.add({"loss": loss,
                               "global_gradients_norm": train_output["global_gradients_norm"],
                               "cross_entropy_loss": train_output["cross_entropy_loss"],
                               "regularization_loss": train_output["regularization_loss"]},
                              step=interactions.value)

                    if args.compute_value:
                        if "errors" in train_output.keys():
                            td_errors = train_output["errors"].numpy()
                        else:
                            assert args.use_value_classification
                            td_errors = batch["returns"] - value_logits_to_scalars(train_output["value_logits"])
                        stats.add({"value_loss": train_output["value_loss"],
                                   "td_error": np.mean(np.abs(td_errors)),
                                   }, step=interactions.value)
            time_learn = time.time() - time_start_learn

            # Evaluate
            if args.eval_episodes > 0:
                if interactions.value - interactions.last_eval_interactions >= args.eval_every_interactions:
                    time_start_eval = time.time()
                    eval_sum_rewards = []
                    eval_steps = []
                    for _ in range(args.eval_episodes):
                        eval_rewards = eval_fn()
                        eval_sum_rewards.append(np.sum(eval_rewards))
                        eval_steps.append(len(eval_rewards))
                    stats.add({"eval_episode_reward": np.mean(eval_sum_rewards),
                               "eval_episode_steps": np.mean(eval_steps),
                               "time_eval": time.time() - time_start_eval},
                              step=interactions.value)
                    stats.report(["eval_episode_reward", "eval_episode_steps"])
                    interactions.last_eval_interactions = interactions.value

            # Statistics
            interactions_step = interactions.value - interactions_before_step
            time_step = time.time() - time_start_step
            stats.add({
                # "nodes_before_plan": nodes_before_plan,
                "nodes_after_plan": nodes_after_plan,
                "nodes_after_execution": nodes_after_execution,

                "generated_nodes": nodes_after_plan - nodes_before_plan,
                "discarded_nodes": nodes_after_plan - nodes_after_execution,
                "delta_nodes": nodes_after_execution - nodes_before_plan,  # generated - discarded
                "interactions_per_step": interactions_step,

                "time_plan": time_plan,
                "time_execute_action": time_execute_action,
                "time_step": time_step,
                "time_learn": time_learn,
                "steps_per_sec": 1/time_step,
                "interactions_per_sec": interactions_step/time_step
                },
                step=interactions.value)

        # Add episode to the dataset
        traj_dict = process_trajectory(trajectory=trajectory,
                                       add_returns=add_returns,
                                       discount_factor=args.discount_factor)
        dataset.extend({k:traj_dict[k] for k in dataset.keys()})  # Add transitions to the buffer that will be used for learning

        stats.add({"episode_reward": sum(traj_dict['rewards']),
                   # "solved_in_one_planning_step": solved_in_one_planning_step,
                   "steps_per_episode": len(traj_dict['rewards']),
                   "memory_usage": memory_usage_fn(),
                   "dataset_size": len(dataset)},
                  step=interactions.value)

        if args.debug:
            stats.add({"reward_found": r_found,
                       "actions_explored": actions_explored,
                       }, step=interactions.value)

        if add_returns:
            stats.add({"return_init_state": traj_dict["returns"][0]}, step=interactions.value)
        if args.compute_value:
            stats.add({"value_init_state": trajectory[0]["v"],
                       "value_init_state_error": np.abs(trajectory[0]["v"] - traj_dict["returns"][0])}, step=interactions.value)

        stats.increment("episodes", step=interactions.value)
        if args.debug:
            report_stats = ["episodes", "episode_reward", "reward_found", "steps_per_episode", "memory_usage", "dataset_size"]
        else:
            report_stats = ["episodes", "episode_reward", "steps_per_episode", "memory_usage", "dataset_size"]
        stats.report(report_stats)
        return trajectory
    return run_episode

def constructor_allow_none(type):
    def constructor(x):
        if x is None or x=="None" or x=="none":
            return None
        return type(x)
    return constructor

def get_log_path(args):
    # LOG PATH
    exp_directory = f"{args.directory}/" if args.directory is not None else ""
    log_path = f"./experiments/{exp_directory}{args.job_id}_{args.env}_{args.low_level_planner}"

    if 'IW' in args.low_level_planner or 'Width' in args.low_level_planner:
        log_path += f"({str(args.low_level_width) if args.low_level_width is not None else 'n' })"

    if args.hierarchical:
        log_path += f"_{args.high_level_planner}"
        if 'IW' in args.high_level_planner or 'Width' in args.high_level_planner:
            log_path += f"({str(args.high_level_width) if args.high_level_width is not None else 'n'})"

    log_path += f"_s{args.seed}"
    return log_path

def generate_hyperparams_file(log_path, job_date, args):
    with open(os.path.join(log_path, "hyperparams.txt"), 'w') as f:
        f.write(job_date + "\n\n")
        print(str(args))
        f.write(str(args))

def get_evaluate_fn(env_eval, preproc_obs_fn, policy_NN, args):
    def _evaluate():
        done = False
        obs = env_eval.reset()
        episode_rewards = []
        while not done:
            x = tf.constant(preproc_obs_fn([obs]).astype(np.float32))
            res = policy_NN(x, training=False)
            p = softmax(res["policy_logits"].numpy().ravel(), temp=args.eval_temp)
            a = sample_pmf(p)
            obs, r, done, info = env_eval.step(a)
            episode_rewards.append(r)
        return episode_rewards
    return _evaluate


def get_value_distr_transformations(min_value, max_value, n_supports):
    assert max_value > min_value
    _range = max_value - min_value
    interval = _range / (n_supports - 1)
    support_vector = np.arange(start=min_value, stop=max_value+interval, step=interval)

    def scalars_to_distributions(values):
        values = np.clip(values, min_value, max_value)

        x = (values - min_value) / interval
        floor_values = np.floor(x).astype(int)
        probs_high = x - floor_values

        d = np.zeros((len(x), n_supports))
        range_x = np.arange(len(x))
        d[range_x, floor_values] = 1 - probs_high
        mask = probs_high > 0 # we'll use this mask to avoid index out of bounds when floor_values are the last index. Also, d already contains 0s.
        d[range_x[mask], floor_values[mask] + 1] = probs_high[mask]
        return d

    def logits_to_scalars(d):
        return np.multiply(softmax(d), support_vector).sum(axis=1)

    return support_vector, scalars_to_distributions, logits_to_scalars



if __name__ == "__main__":
    import gridenvs.examples  # load simple envs
    import envs  # load L and XL envs
    import pddl2gym.blocks  # import registers gym environments
    import pddl2gym.blocks_columns


    logger.setLevel(logging.INFO)

    args = Params().parse_args()

    planners = {p_class.__name__: p_class for p_class in [RolloutIW, BFS, IW, CountbasedRolloutIW]}
    low_level_planner_class = planners[args.low_level_planner]
    high_level_planner_class = planners[args.high_level_planner] if args.hierarchical else None

    log_path = get_log_path(args)
    os.makedirs(log_path)
    generate_hyperparams_file(log_path, job_date, args)

    # if args.compute_value or args.guide_plan_network_policy:
    #     assert args.learn

    if args.use_value:
        assert args.compute_value

    if args.guide_plan_network_policy:
        network_policy = lambda node: node.data["probs"]  # Policy to guide the planner: NN output probabilities
    else:
        network_policy = None

    use_network = args.guide_plan_network_policy or args.compute_value or args.features == "dynamic" or args.learn
    args.replay_min_transitions = min(args.replay_min_transitions, args.replay_capacity)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create counter
    interactions = InteractionsCounter(budget=args.interactions_budget)

    # Create env
    add_downsampling = args.hierarchical
    env, preproc_obs_fn = make_env(args.env, args.max_episode_steps, add_downsampling=add_downsampling,
                                   downsampling_tiles_w=args.downsampling_tiles_w,
                                   downsampling_tiles_h=args.downsampling_tiles_h,
                                   downsampling_pix_values=args.downsampling_pix_values,
                                   atari_frameskip=args.atari_frameskip)

    observe_fns = []
    observe_fns.append(lambda node: interactions.increment())
    if args.features == "BASIC":
        gridenvs_BASIC_features = get_gridenvs_BASIC_features_fn(env, features_name="low_level_features")
        observe_fns.append(gridenvs_BASIC_features)

    if args.hierarchical:
        high_level_feats_fn = get_downsampled_features_fn(env, features_name="high_level_features")
        observe_fns.append(high_level_feats_fn)

    if use_network:
        # Define model
        model = Mnih2013(dense_units=args.model_dense_units,
                         num_logits=env.action_space.n,
                         add_value=args.compute_value,
                         num_value_logits=(args.value_classification_supports if args.use_value_classification else 1),
                         use_batch_normalization=args.use_batch_normalization,
                         use_batch_renorm=args.use_batch_renorm)

        call_model = tf.function(model, autograph=False) if args.use_graph else model.__call__

        if args.compute_value and args.use_value_classification:
            support_vector, value_scalars_to_distrs, value_logits_to_scalars = get_value_distr_transformations(min_value=args.value_classification_min,
                                                                                                               max_value=args.value_classification_max,
                                                                                                               n_supports=args.value_classification_supports)
        else:
            value_scalars_to_distrs = value_logits_to_scalars = None

        # Define callback function for new observations (it will be called after each environment interaction)
        observe_nn_fn = get_observe_nn_fn(model=call_model,
                                          preproc_obs_fn=preproc_obs_fn,
                                          get_features=(args.features == "dynamic"),
                                          args=args,
                                          value_logits_to_scalars=value_logits_to_scalars)
        observe_fns.append(observe_nn_fn)

    # TreeActor provides equivalent functions to env.step() and env.reset() for on-line planning: it creates a tree,
    # adds nodes to it and allows us to take steps (maybe keeping the subtree)
    env_actions = list(range(env.action_space.n))
    low_level_actor = EnvTreeActor(env,
                                   observe_fns=observe_fns,
                                   applicable_actions_fn=lambda n: env_actions)

    # Define planner
    if low_level_planner_class is RolloutIW:
        if args.use_features_nt:
            assert args.features == "dynamic"
            n_features = args.model_dense_units
            n_values = 2
        else:
            n_features = n_values = None
        low_level_planner = RolloutIW(generate_successor_fn=low_level_actor.generate_successor,
                                      policy_fn=network_policy, branching_factor=env.action_space.n,
                                      width=args.low_level_width, features_name="low_level_features",
                                      ensure_same_initialization=args.RIW_ensure_same_init,
                                      ignore_terminal_nodes=True,
                                      n_features=n_features, n_values=n_values)
    elif low_level_planner_class is IW:
        low_level_planner = IW(generate_successor_fn=low_level_actor.generate_successor,
                               width=args.low_level_width, features_name="low_level_features",
                               ignore_terminal_nodes=True)
    elif low_level_planner_class is CountbasedRolloutIW:
        low_level_planner = CountbasedRolloutIW(generate_successor_fn=low_level_actor.generate_successor,
                                                width=args.low_level_width, features_name="low_level_features",
                                                temp=args.countbasedRIW_temp, ignore_terminal_nodes=True)
    elif low_level_planner_class is BFS:
        low_level_planner = BFS(generate_successor_fn=low_level_actor.generate_successor,
                                features_name="low_level_features")
    else:
        raise ValueError(f"Low level planner class {low_level_planner_class.__name__} not known.")

    low_level_planner.add_stop_fn(lambda tree: not interactions.within_budget())
    low_level_planner.add_stop_fn(lambda tree: low_level_actor.get_tree_size(tree) >= args.max_tree_size)

    if args.guide_plan_network_policy:
        assert low_level_planner_class is RolloutIW
    if low_level_planner_class is RolloutIW:
        assert args.guide_plan_network_policy


    high_level_planner = None
    high_level_actor = None
    if args.hierarchical:
        high_level_actor = AbstractTreeActor(low_level_planner, low_level_actor)

        if high_level_planner_class is IW:
            high_level_planner = IW(generate_successor_fn=high_level_actor.generate_successor,
                                    width=args.high_level_width, features_name="high_level_features",
                                    ignore_terminal_nodes=True)
        elif high_level_planner_class is CountbasedRolloutIW:
            high_level_planner = CountbasedRolloutIW(generate_successor_fn=high_level_actor.generate_successor,
                                                     width=args.high_level_width, features_name="high_level_features",
                                                     temp=args.countbasedRIW_temp, ignore_terminal_nodes=True)
        elif high_level_planner_class is BFS:
            high_level_planner = BFS(generate_successor_fn=high_level_actor.generate_successor,
                                     features_name="high_level_features")
        else:
            raise ValueError(f"High level planner class {high_level_planner_class.__name__} not known.")

        high_level_planner.add_stop_fn(lambda tree: not interactions.within_budget())
        high_level_planner.add_stop_fn(lambda tree: high_level_actor.get_tree_size(tree) >= args.max_tree_size)

    train_fn = None
    if args.learn:
        loss_fn = get_loss_fn(model, args)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate,
                                                rho=args.rmsprop_decay,
                                                epsilon=args.rmsprop_epsilon)
        train_fn = get_train_fn(model=model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                max_grad_norm=args.max_grad_norm,
                                use_graph=args.use_graph)

    eval_fn = None
    if args.eval_episodes > 0 and use_network:
        env_eval, _ = make_env(args.env, args.max_episode_steps, add_downsampling=False,
                               downsampling_tiles_w=None, downsampling_tiles_h=None,
                               downsampling_pix_values=None,
                               atari_frameskip=args.atari_frameskip)
        eval_fn = get_evaluate_fn(env_eval=env_eval,
                                  preproc_obs_fn=preproc_obs_fn,
                                  policy_NN=call_model,
                                  args=args)



    process = psutil.Process()
    memory_usage_fn = lambda: process.memory_info().rss

    stats = Stats(use_tensorboard=args.use_tensorboard, log_path=log_path)
    experience_keys = ["observations", "target_policy"]
    if args.compute_value:
        experience_keys.append("returns")

    experience_replay = ExperienceReplay(keys=experience_keys,
                                         capacity=args.replay_capacity)

    run_episode_fn = get_episode_fn(actor=high_level_actor if args.hierarchical else low_level_actor,
                                    planner=high_level_planner if args.hierarchical else low_level_planner,
                                    train_fn=train_fn,
                                    dataset=experience_replay,
                                    add_returns=args.compute_value,
                                    stats=stats,
                                    memory_usage_fn=memory_usage_fn,
                                    preproc_obs_fn=preproc_obs_fn,
                                    eval_fn=eval_fn,
                                    n_actions=env.action_space.n,
                                    value_scalars_to_distrs=value_scalars_to_distrs,
                                    value_logits_to_scalars=value_logits_to_scalars,
                                    args=args)

    # MAIN LOOP
    last_save_interactions = 0
    try:
        # Initialize experience replay: run complete episodes until we exceed both batch_size and dataset_min_transitions
        low_level_planner.set_policy_fn(None)
        print("Initializing experience replay", flush=True)
        while len(experience_replay) < args.batch_size or len(experience_replay) < args.replay_min_transitions:
            run_episode_fn(train=False,
                           use_value_for_tree_policy=args.use_value and args.use_value_at_init)
        if interactions.value - last_save_interactions >= args.save_every_interactions:
            stats.save('stats.h5')
            last_save_interactions = interactions.value
            if args.save_network:
                model.save_weights(os.path.join(log_path, "model_weights.h5"))

        # Interleave planning and learning steps
        low_level_planner.set_policy_fn(network_policy)
        print("\nInterleaving planning and learning steps.", flush=True)
        while interactions.value < args.max_interactions:
            run_episode_fn(train=(train_fn is not None),
                           use_value_for_tree_policy=args.use_value)


            if interactions.value - last_save_interactions >= args.save_every_interactions:
                stats.save('stats.h5')
                last_save_interactions = interactions.value
                if args.save_network:
                    model.save_weights(os.path.join(log_path, "model_weights.h5"))

    finally:
        stats.save("stats.h5")
        if args.save_network:
            model.save_weights(os.path.join(log_path, "model_weights.h5"))

