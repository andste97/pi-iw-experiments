import random

import hydra
import numpy as np
import torch

import wandb

from models.model_dynamic import DQNDynamic
from datetime import datetime
import gym
import gridenvs.examples  # register GE environments to gym

import pytorch_lightning as pl


@hydra.main(
    config_path="models/config",
    config_name="config_atari_dynamic.yaml",
    version_base="1.3",
)
def main(config):
    # set seeds, numpy for planner, torch for policy
    seed_everything(config.train.seed)

    run = wandb.init(
        project="pi-iw-experiments-piiw",
        id=f'{config.train.env_id.replace("ALE/", "")}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")}',
        #mode='offline',
    )


    # choose wither to uses dynamic or BASIC features
    model = DQNDynamic(config)
    model.fit()

    model.test_model()

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    main()
