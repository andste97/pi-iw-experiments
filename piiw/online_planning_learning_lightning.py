import hydra
import numpy as np
import torch.optim
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from piiw.models.lightning_model_basic import LightningDQN
from piiw.models.lightning_model_dynamic import LightningDQNDynamic
import timeit
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
    frametime = 1  # in milliseconds to display renderings

    nodes_generated = []
    times = []
    rewards = []
    start_time = timeit.default_timer()

    # set seeds, numpy for planner, torch for policy
    pl.seed_everything(config.train.seed)

    logger = WandbLogger(
        log_model=True, project="pi-iw-experiments-piiw",
        id=f'{config.train.env_id.replace("ALE/", "")}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")}'
    )


    # choose wither to uses dynamic or BASIC features
    if config.model.use_dynamic_features:
        model = LightningDQNDynamic(config)
    else:
        model = LightningDQN(config)
    logger.watch(model)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=config.train.max_epochs,
        logger=logger,
        deterministic="warn",
        enable_checkpointing=True
    )

    trainer.fit(
        model
    )
    # save logged data
    logger.save()


if __name__ == "__main__":
    main()
