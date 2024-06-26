# This file is for additional tests on the policy.
# Instead of using the planner, this one uses

from collections import OrderedDict

import omegaconf
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from atari_utils.make_env import make_env
from models.mnih_2013 import Mnih2013
import numpy as np
import wandb

from data.experience_replay import ExperienceReplay
from planners.rollout_IW import RolloutIW
from tree_utils.tree_actor import EnvTreeActor
from utils.interactions_counter import InteractionsCounter
from utils.utils import softmax, sample_pmf, reward_in_tree, display_image_cv2
from utils.pytorch_utils import configure_optimizer_based_on_config
import PIL

class LightningDQNTest(pl.LightningModule):
    """The model used by MNIH 2013 paper of DQN."""
    def __init__(self,
                 config: omegaconf.DictConfig):
        super().__init__()

        if (not OmegaConf.is_config(config)):
            config = OmegaConf.create(config)

        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config))

        self.env = make_env(config.train.env_id, config.train.episode_length,
                            atari_frameskip=config.train.atari_frameskip)
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
            num_logits=self.env.action_space.n,
            add_value=config.model.add_value,
            use_dynamic_features=config.model.use_dynamic_features
        )

        self.model = model.to(self.device)

        self.experience_replay = ExperienceReplay(
            capacity=config.train.replay_capacity,
            keys=config.train.experience_keys
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        # intialize planner

        self.interactions = InteractionsCounter(budget=config.plan.interactions_budget)
        self.total_interactions = InteractionsCounter(budget=config.train.total_interaction_budget)

        # create observe functions
        increase_interactions_fn = lambda node: self.interactions.increment()
        increase_total_interactions_fn = lambda node: self.total_interactions.increment()

        self.actor = EnvTreeActor(
            self.env,
            observe_fns=[
                self.get_compute_dynamic_policy_output_fn(model),
                increase_interactions_fn,
                increase_total_interactions_fn
            ],
            applicable_actions_fn=self.applicable_actions_fn
        )

        self.planner = RolloutIW(
            policy_fn=network_policy,
            generate_successor_fn=self.actor.generate_successor,
            width=config.plan.width,
            branching_factor=self.env.action_space.n
        )

        self.planner.add_stop_fn(lambda tree: not self.interactions.within_budget() or reward_in_tree(tree))

        self.tree = self.actor.reset()
        self.episode_step = 0
        self.episodes = 0
        self.episode_reward = 0
        self.episode_done = False

    def configure_optimizers(self):
        """ Initialize optimizer"""
        optimizer = configure_optimizer_based_on_config(self.model, self.config)
        return [optimizer]


    def applicable_actions_fn(self):
        env_actions = list(range(self.env.action_space.n))
        return env_actions


    def get_compute_dynamic_policy_output_fn(self, model):
        def dynamic_policy_output(node):
            x = torch.tensor(np.expand_dims(node.data["obs"], axis=0), dtype=torch.float32,
                             device=self.device)

            with torch.no_grad():
                logits, features = model(x)
            node.data["probs"] = softmax(np.array(logits.to("cpu").ravel()), temp=self.config.plan.softmax_temperature)
            node.data["features"] = list(
                enumerate(features.to("cpu").numpy().ravel().astype(bool)))  # discretization -> bool

        return dynamic_policy_output

    def test_model(self):
        tree = self.actor.reset()
        episode_rewards = 0

        images = []
        feature_list = []

        for i in tqdm(range(self.config.train.episode_length), desc="Running tests"):
            node = tree.root

            x = torch.tensor(np.expand_dims(node.data["obs"], axis=0), dtype=torch.float32,
                             device=self.device)


            with torch.no_grad():
                logits, features = self.model(x)
            #node.data["probs"] = softmax(np.array(logits.to("cpu").ravel()), temp=0)
            #node.data["features"] = list(enumerate(features.to("cpu").numpy().ravel().astype(bool)))

            probs = softmax(np.array(logits.to("cpu").ravel()), temp=0)
            features = list(enumerate(features.to("cpu").numpy().ravel().astype(bool)))
            feature_list.append(features)

            step_action = sample_pmf(probs)
            self.actor.generate_successor(tree, node, step_action)
            prev_root_data, node = self.actor.step(tree, step_action, cache_subtree=False)


            images.append(self.actor.render(tree, size=(200,200)))
            episode_rewards += node.data["r"]
            wandb.log({'test/rewards': episode_rewards})

            if node.data["done"]:
                wandb.log({'test/episode_steps': i})
                break

        result = []
        for featurel in feature_list:
            result.append([int(feature[1]) for feature in featurel])
        result = np.array(result)
        result *= 255
        result = np.transpose(result) # make steps go from left to right. I.e. leftmost column -> 1. step, rightmost -> last step
        wandb.log({"test/video": wandb.Video(np.array(images), fps=5)})
        wandb.log({"test/features": wandb.Image(result, caption="some caption")})


def planning_step(actor,
                  planner,
                  interactions,
                  dataset,
                  tree,
                  cache_subtree,
                  discount_factor,
                  n_action_space,
                  softmax_temp
                  ):
    interactions.reset_budget()
    planner.initialize(tree=tree)
    planner.plan(tree=tree)

    actor.compute_returns(tree, discount_factor=discount_factor, add_value=False)

    step_Q = sample_best_action(node=tree.root,
                                n_actions=n_action_space,
                                discount_factor=discount_factor)

    policy_output = softmax(step_Q, temp=softmax_temp)
    step_action = sample_pmf(policy_output)

    prev_root_data, current_root = actor.step(tree, step_action, cache_subtree=cache_subtree)

    tensor_pytorch_format = torch.tensor(np.array(prev_root_data["obs"]), dtype=torch.float32)

    dataset.append({"observations": tensor_pytorch_format,
                    "target_policy": torch.tensor(policy_output, dtype=torch.float32)})
    return current_root.data["r"], current_root.data["done"]


def network_policy(node):
    return node.data["probs"]


def sample_best_action(node, n_actions, discount_factor):
    Q = np.empty(n_actions, dtype=np.float32)
    Q.fill(-np.inf)
    for child in node.children:
        Q[child.data["a"]] = child.data["r"] + discount_factor * child.data["R"]
    return Q