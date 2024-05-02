from collections import OrderedDict

import omegaconf
from omegaconf import OmegaConf
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

from atari_utils.make_env import make_env
from data.test_dummy_dataset import DummyDataset
from models.mnih_2013 import Mnih2013
import numpy as np
import wandb

from data.ExperienceDataset import ExperienceDataset
from data.experience_replay import ExperienceReplay
from planners.rollout_IW import RolloutIW
from tree_utils.tree_actor import EnvTreeActor
from utils.interactions_counter import InteractionsCounter
from utils.utils import softmax, sample_pmf, reward_in_tree, display_image_cv2
from utils.pytorch_utils import configure_optimizer_based_on_config
from visualization.visualize_tree_with_observations import visualize_tree_with_observations
import sys
import os


class LightningDQNDynamic(pl.LightningModule):
    """The model used by MNIH 2013 paper of DQN."""

    def __init__(self,
                 config: omegaconf.DictConfig):
        super().__init__()

        if (not OmegaConf.is_config(config)):
            config = OmegaConf.create(config)

        assert config.model.use_dynamic_features

        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config))

        self.env = make_env(config.train.env_id, config.train.episode_length,
                            atari_frameskip=config.train.atari_frameskip)
        # set env seed
        self.env.seed(config.train.seed)

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
            branching_factor=self.env.action_space.n,
            ignore_cached_nodes=True
        )

        self.planner.add_stop_fn(lambda tree: not self.interactions.within_budget())

        self.tree = self.actor.reset()
        self.episode_step = 0
        self.episodes = 0
        self.aux_replay = []
        self.best_episode_reward = -sys.maxsize - 1
        self.initialize_experience_replay(config.train.replay_init_capacity)
        self.episode_reward = 0
        self.episode_done = False

    def configure_optimizers(self):
        """ Initialize optimizer"""
        optimizer = configure_optimizer_based_on_config(self.model, self.config)
        return [optimizer]

    def on_train_batch_start(self, batch, batch_idx):
        """This method gets called before each training step
        If -1 is returned, the current epoch is ended
        We use this to tie the epochs together with the episodes
        of the simulator.
        See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html#lightning.pytorch.core.hooks.ModelHooks.on_train_batch_start"""
        if self.episode_done: # simulator episode done, end current epoch
            self.episode_done = False
            return -1

    def training_step(self, batch, batch_idx):
        observations, target_policy = batch

        r, self.episode_done = self.planning_step(
            actor=self.actor,
            planner=self.planner,
            interactions=self.interactions,
            aux_replay=self.aux_replay,
            tree=self.tree,
            cache_subtree=self.config.plan.cache_subtree,
            discount_factor=self.config.plan.discount_factor,
            n_action_space=self.env.action_space.n,
            softmax_temp=self.config.plan.softmax_temperature
        )
        self.episode_reward += r

        logits, features = self.model(observations)
        loss = self.criterion(logits, target_policy)

        # end of episode/epoch logging needs to go here
        # because it doesn't get logged in on_train_batch_start
        if self.episode_done:
            if self.episode_reward >= self.best_episode_reward:
                self.best_episode_reward = self.episode_reward
                self.experience_replay.append_all(self.aux_replay)
            self.reset_aux_replay()
            self.log_dict({'train/episode': float(self.episodes),
                           'train/episode_steps': float(self.episode_step),
                           'train/episode_reward': self.episode_reward,})
            self.logger.save()
            self.tree = self.actor.reset()
            self.episodes += 1
            self.episode_step = 0
            self.episode_reward = 0

        # Log loss and metric
        self.log_dict({"train/loss": loss,
                       'total_interactions': float(self.total_interactions.value)})

        self.episode_step += 1
        return OrderedDict({'loss': loss})

    def applicable_actions_fn(self):
        env_actions = list(range(self.env.action_space.n))
        return env_actions

    def initialize_experience_replay(self, warmup_length):
        pbar = tqdm(total=warmup_length, desc="Initializing experience replay")

        # make sure we cannot get stuck in infinite loop
        assert self.config.train.replay_capacity >= warmup_length

        while len(self.aux_replay) < warmup_length:
            cur_length = len(self.aux_replay)
            r, episode_done = self.planning_step(
                actor=self.actor,
                planner=self.planner,
                interactions=self.interactions,
                aux_replay=self.aux_replay,
                tree=self.tree,
                cache_subtree=self.config.plan.cache_subtree,
                discount_factor=self.config.plan.discount_factor,
                n_action_space=self.env.action_space.n,
                softmax_temp=self.config.plan.softmax_temperature
            )
            # self.actor.render(size=(800, 800), tree=self.tree)
            pbar.update(len(self.aux_replay) - cur_length)
            if episode_done:
                self.tree = self.actor.reset()

        self.experience_replay.append_all(self.aux_replay)
        self.reset_aux_replay()
        self.tree = self.actor.reset()

    def reset_aux_replay(self):
        self.aux_replay = []

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceDataset(self.experience_replay, self.config.train.batch_size,
                                    self.config.train.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.train.batch_size,
        )
        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = DummyDataset(1)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1
        )

        return dataloader

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

    def validation_step(self, batch, batch_idx):
        tree = self.actor.reset()
        episode_rewards = []

        test_results = ExperienceReplay(
            capacity=self.config.train.replay_capacity,
            keys=self.config.train.experience_keys
        )

        images = []

        img = self.env.render(mode='rgb_array')
        img = np.transpose(img, axes=[2, 0, 1])
        images.append(img)

        for i in tqdm(range(self.config.train.episode_length), desc="Running tests"):
            r, episode_done = self.planning_step(
                actor=self.actor,
                planner=self.planner,
                interactions=self.interactions,
                aux_replay=[],
                tree=tree,
                cache_subtree=self.config.plan.cache_subtree,
                discount_factor=self.config.plan.discount_factor,
                n_action_space=self.env.action_space.n,
                softmax_temp=self.config.plan.softmax_temperature
            )

            # set env to root tree, otherwise env is set to last visited state in planner
            self.env.restore_state(tree.root.data["s"])

            # transform and add rgb image to output
            img = self.env.render(mode='rgb_array')
            img = np.transpose(img, axes=[2,0,1])
            images.append(img)
            episode_rewards.append(r)
            wandb.log({'test/rewards': episode_rewards[-1]})

            if episode_done:
                wandb.log({'test/episode_steps': i})
                break

        wandb.log({"test/video": wandb.Video(np.array(images), fps=5, format="mp4")})
        wandb.log({"test/total_test_reward": np.sum(episode_rewards)})
        return OrderedDict({'testing_rewards': episode_rewards})

    def planning_step(self,
                      actor,
                      planner,
                      interactions,
                      aux_replay,
                      tree,
                      cache_subtree,
                      discount_factor,
                      n_action_space,
                      softmax_temp,
                      should_visualize=False
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

        if should_visualize:
            from visualization.visualize_tree_with_observations import visualize_tree_with_observations
            visualize_tree_with_observations(tree.root, f'../reports/output_steps/ep_{self.episodes}_step_{self.episode_step}.png')

        prev_root_data, current_root = actor.step(tree, step_action, cache_subtree=cache_subtree)

        tensor_pytorch_format = torch.tensor(np.array(prev_root_data["obs"]), dtype=torch.float32)

        aux_replay.append({"observations": tensor_pytorch_format,
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
