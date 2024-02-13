import hydra
import numpy as np
import torch.optim
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger

from piiw.models.pytorch_model import LightningDQN
import timeit
import gym
import gridenvs.examples  # register GE environments to gym

import pytorch_lightning as pl


@hydra.main(
    config_path="models/config",
    config_name="config.yaml",
    version_base="1.3",
)
def main(config):
    frametime = 1  # in milliseconds to display renderings

    nodes_generated = []
    times = []
    rewards = []
    start_time = timeit.default_timer()

    # set seeds, numpy for planner, torch for policy
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    logger = WandbLogger(log_model=True)

    wandb.config.update(OmegaConf.to_container(config))

    model = LightningDQN(config)

    trainer = pl.Trainer(
        max_epochs=10000,
        logger=logger
    )

    trainer.fit(model)
    # save logged data
    logger.save()


if __name__ == "__main__":
    wandb.init()
    main()
