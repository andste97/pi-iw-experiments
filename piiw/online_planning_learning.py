import hydra
import numpy as np
import torch.optim

from data.experience_replay import ExperienceReplay
from utils.utils import sample_pmf, reward_in_tree
from utils.utils import softmax
from utils.interactions_counter import InteractionsCounter
from planners.rollout_IW import RolloutIW
import timeit
import gym
from tree_utils.tree_actor import EnvTreeActor
from models.mnih_2013 import Mnih2013
import gridenvs.examples  # register GE environments to gym


def get_gridenvs_BASIC_features_fn(env):
    def gridenvs_BASIC_features(node):
        node.data["features"] = tuple(enumerate(env.unwrapped.get_char_matrix().flatten()))  # compute BASIC features

    return gridenvs_BASIC_features

def get_compute_policy_output_fn(model):
    def policy_output(node):
        # it's much faster to compute the policy output when expanding the state
        # than every time the planner tries to accesss the state (which is what the
        # happens when you put the model answer into the network_policy fn)
        x = torch.tensor(np.expand_dims(node.data["obs"], axis=0), dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            logits = model(x)
        node.data["probs"] = np.array(torch.nn.functional.softmax(logits[0], dim=1).data).ravel()

    return policy_output


# Function that will be executed at each interaction with the environment
# def observe_pi_iw_dynamic(model, node):
#     x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
#     logits, features = model(x, use_dynamic_features=True)
#     node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
#     node.data["features"] = features_to_atoms(features.numpy().ravel().astype(np.bool)) # discretization -> bool


def network_policy(node):
        return node.data["probs"]


def sample_best_action(node, n_actions, discount_factor):
    Q = np.empty(n_actions, dtype=np.float32)
    Q.fill(-np.inf)
    for child in node.children:
        Q[child.data["a"]] = child.data["r"] + discount_factor * child.data["R"]
    return Q


def planning_step(actor,
                  planner,
                  interactions,
                  dataset,
                  tree,
                  cache_subtree,
                  discount_factor,
                  n_action_space
                  ):
    interactions.reset_budget()
    planner.initialize(tree=tree)
    planner.plan(tree=tree)

    actor.compute_returns(tree, discount_factor=discount_factor, add_value=False)

    step_Q = sample_best_action(node=tree.root,
                                n_actions=n_action_space,
                                discount_factor=discount_factor)

    policy_output = softmax(step_Q, temp=0)
    step_action = sample_pmf(policy_output)

    #print("action: ", step_action)
    prev_root_data, current_root = actor.step(tree, step_action, cache_subtree=cache_subtree)

    tensor_pytorch_format = torch.tensor(prev_root_data["obs"], dtype=torch.float32)
    tensor_pytorch_format = tensor_pytorch_format.permute(2, 0, 1).contiguous()

    dataset.append({"observations": tensor_pytorch_format,
                    "target_policy": torch.tensor(policy_output, dtype=torch.float32)})
    return current_root.data["r"], current_root.data["done"]


@hydra.main(
    config_path="models/config",
    config_name="config_basic.yaml",
    version_base="1.3",
)
def action(config):
    frametime = 1  # in milliseconds to display renderings

    nodes_generated = []
    times = []
    rewards = []
    start_time = timeit.default_timer()

    # set seeds, numpy for planner, torch for policy
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
    env = gym.make(config.train.env_id)

    model = Mnih2013(
        conv1_in_channels=config.model.conv1_in_channels,
        conv1_out_channels=config.model.conv1_out_channels,
        conv1_kernel_size=config.model.conv1_kernel_size,
        conv1_stride=config.model.conv1_stride,
        conv2_out_channels=config.model.conv2_out_channels,
        conv2_kernel_size=config.model.conv2_kernel_size,
        conv2_stride=config.model.conv2_stride,
        fc1_in_features=config.model.fc1_in_features,
        fc1_out_features=config.model.fc1_out_features,
        num_logits=env.action_space.n,
        add_value=config.model.add_value
    )

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config.train.learning_rate,
        alpha=config.train.rmsprop_alpha,  # same as rho in tf
        eps=config.train.rmsprop_epsilon,
        weight_decay=config.train.l2_reg_factor
        # use this for l2 regularization to replace TF regularization implementation
    )

    experience_replay = ExperienceReplay(
        capacity=config.train.replay_capacity,
        keys=config.train.experience_keys
    )

    criterion = torch.nn.CrossEntropyLoss()

    interactions = InteractionsCounter(budget=config.plan.interactions_budget)
    total_interactions = InteractionsCounter(budget=config.train.total_interaction_budget)

    # create observe functions
    increase_interactions_fn = lambda node: interactions.increment()
    increase_total_interactions_fn = lambda node: total_interactions.increment()

    env_actions = list(range(env.action_space.n))
    applicable_actions_fn = lambda n: env_actions

    actor = EnvTreeActor(env,
                         observe_fns=[
                             get_gridenvs_BASIC_features_fn(env),
                             get_compute_policy_output_fn(model),
                             increase_interactions_fn,
                             increase_total_interactions_fn
                         ],
                         applicable_actions_fn=applicable_actions_fn)

    planner = RolloutIW(
        policy_fn=network_policy,
        generate_successor_fn=actor.generate_successor,
        width=config.plan.width,
        branching_factor=env.action_space.n
    )
    # planner = IW(generate_successor_fn=actor.generate_successor, width=width, ignore_terminal_nodes=True)
    # planner = BFS(generate_successor_fn=actor.generate_successor)
    # planner = CountbasedRolloutIW(generate_successor_fn=actor.generate_successor, width=width)

    tree = actor.reset()
    planner.add_stop_fn(lambda tree: not interactions.within_budget() or reward_in_tree(tree))

    # filling up the experience dataset
    print("Initializing experience replay", flush=True)
    assert config.train.replay_capacity >= config.train.batch_size
    assert config.train.replay_capacity >= config.train.episode_length
    while len(experience_replay) < config.train.batch_size:
        r, episode_done = planning_step(
            actor=actor,
            planner=planner,
            interactions=interactions,
            dataset=experience_replay,
            tree=tree,
            cache_subtree=config.plan.cache_subtree,
            discount_factor=config.plan.discount_factor,
            n_action_space=env.action_space.n
        )
        if episode_done:
            tree = actor.reset()

    print("beginning training")
    model.train()
    while total_interactions.within_budget():
        steps_cnt = 0
        tree = actor.reset()
        episode_done = False
        while not episode_done:
            r, episode_done = planning_step(
                actor=actor,
                planner=planner,
                interactions=interactions,
                dataset=experience_replay,
                tree=tree,
                cache_subtree=config.plan.cache_subtree,
                discount_factor=config.plan.discount_factor,
                n_action_space=env.action_space.n
            )

            optimizer.zero_grad()
            _, batch = experience_replay.sample(config.train.batch_size)

            # tensors were created for tensorflow, which has channel-last input shape
            # but pytorch has channel-first input shape.
            observations = batch["observations"]
            target_policy = batch["target_policy"]

            logits = model(observations)
            loss = criterion(logits[0], target_policy)
            loss.backward()
            optimizer.step()

            steps_cnt += 1
        print("Episode done. Steps count: ", steps_cnt, "Episode number:")
        print("environment interactions: ", total_interactions.value)


if __name__ == "__main__":
    action()
