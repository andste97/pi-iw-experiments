from collections import OrderedDict

import omegaconf
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

from atari_utils.make_env import make_env
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
from atari_utils.atari_wrappers import is_atari_env


class LightningDQNDynamic(pl.LightningModule):
    """The model used by MNIH 2013 paper of DQN."""

    def __init__(self,
                 config: omegaconf.DictConfig):
        super().__init__()

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
        self.initialize_experience_replay(self.config.train.batch_size)
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

        r, self.episode_done = planning_step(
            actor=self.actor,
            planner=self.planner,
            interactions=self.interactions,
            dataset=self.experience_replay,
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

        while len(self.experience_replay) < warmup_length:
            cur_length = len(self.experience_replay)
            r, episode_done = planning_step(
                actor=self.actor,
                planner=self.planner,
                interactions=self.interactions,
                dataset=self.experience_replay,
                tree=self.tree,
                cache_subtree=self.config.plan.cache_subtree,
                discount_factor=self.config.plan.discount_factor,
                n_action_space=self.env.action_space.n,
                softmax_temp=self.config.plan.softmax_temperature
            )
            # self.actor.render(size=(800, 800), tree=self.tree)
            pbar.update(len(self.experience_replay) - cur_length)
            if episode_done:
                self.tree = self.actor.reset()

        self.tree = self.actor.reset()

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceDataset(self.experience_replay, self.config.train.batch_size,
                                    self.config.train.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.train.batch_size,
        )
        return dataloader

    def get_compute_dynamic_policy_output_fn(self, model):
        def dynamic_policy_output(node):
            x = torch.tensor(np.expand_dims(node.data["obs"], axis=0), dtype=torch.float32,
                             device=self.device)

            with torch.no_grad():
                logits, features = model(x)
                # Todo: check if pytorch softmax breaks something. After all it uses temp=1
            #node.data["probs"] = np.array(torch.nn.functional.softmax(logits, dim=1).to("cpu").data).ravel()
            node.data["probs"] = softmax(np.array(logits.to("cpu").ravel()), temp=self.config.plan.softmax_temperature)
            node.data["features"] = list(
                enumerate(features.to("cpu").numpy().ravel().astype(bool)))  # discretization -> bool

        return dynamic_policy_output

    def test_model(self):
        tree = self.actor.reset()
        episode_rewards = 0

        #test_interactions = InteractionsCounter(budget=self.config.plan.interactions_budget)
        test_results = ExperienceReplay(
            capacity=self.config.train.replay_capacity,
            keys=self.config.train.experience_keys
        )

        images = []

        for i in tqdm(range(self.config.train.episode_length), desc="Running tests"):
            r, episode_done = planning_step(
                actor=self.actor,
                planner=self.planner,
                interactions=self.interactions,
                dataset=test_results,
                tree=tree,
                cache_subtree=self.config.plan.cache_subtree,
                discount_factor=self.config.plan.discount_factor,
                n_action_space=self.env.action_space.n,
                softmax_temp=self.config.plan.softmax_temperature
            )
            images.append(self.actor.render(tree, size=(200,200)))
            episode_rewards += r
            wandb.log({'test/rewards': episode_rewards})

            if episode_done:
                wandb.log({'test/episode_steps': i})
                break

        wandb.log({"test/video": wandb.Video(np.array(images), fps=5)})
        return OrderedDict({'testing_rewards': episode_rewards})



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
