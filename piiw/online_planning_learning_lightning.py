import hydra
import numpy as np
import torch.optim
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from piiw.models.lightning_model_basic import LightningDQN
from piiw.models.lightning_model_dynamic import LightningDQNDynamic
import timeit
import gym
import gridenvs.examples  # register GE environments to gym

import pytorch_lightning as pl


@hydra.main(
    config_path="models/config",
    config_name="config_dynamic.yaml",
    version_base="1.3",
)
def main(config):
    run = wandb.init(config=OmegaConf.to_container(config))

    frametime = 1  # in milliseconds to display renderings

    nodes_generated = []
    times = []
    rewards = []
    start_time = timeit.default_timer()

    # set seeds, numpy for planner, torch for policy
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    logger = WandbLogger(log_model=True)

    model = LightningDQNDynamic(config)
    logger.watch(model)

    trainer = pl.Trainer(
        accelerator="cuda",
        max_epochs=1000,
        logger=logger
    )

    trainer.fit(model)
    # save logged data
    logger.save()


if __name__ == "__main__":
    main()
