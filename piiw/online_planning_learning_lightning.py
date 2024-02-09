import hydra
import numpy as np
import torch.optim

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

    model = LightningDQN(config)

    trainer = pl.Trainer(
        max_epochs=10000
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
