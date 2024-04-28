from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import wandb
from pytorch_lightning.loggers import WandbLogger

from models.lightning_model_basic import LightningDQN
from models.lightning_model_dynamic import LightningDQNDynamic
import timeit
from datetime import datetime
import gym
import gridenvs.examples  # register GE environments to gym

import pytorch_lightning as pl

from models.lightning_model_evaluator import LightningDQNTest


@hydra.main(
    config_path="models/config",
    config_name="config_dynamic_explanatory_test.yaml",
    version_base="1.3",
)
def main(config):
    eval_checkpoint_name = 'piiw-thesis/pi-iw-experiments-piiw/model-GE_MazeKeyDoor-v2_2024-04-10_11-07-45.212123:v14'

    # set seeds, numpy for planner, torch for policy
    pl.seed_everything(config.train.seed)
    #torch.use_deterministic_algorithms((True))

    run = wandb.init(
        project="pi-iw-experiments-piiw",
        id=f'TEST_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")}',
        notes=f'evaluating run for checkpoint: {eval_checkpoint_name}'
        #offline=True,
        #log_model=False # needs to be False when offline is enabled
    )

    artifact = run.use_artifact(eval_checkpoint_name, type='model')
    artifact_dir = artifact.download()
    checkpoint_path = Path(artifact_dir) / "model.ckpt"

    # choose wither to uses dynamic or BASIC features
    if config.model.use_dynamic_features:
        model = LightningDQNTest.load_from_checkpoint(checkpoint_path)
    else:
        model = LightningDQN.load_from_checkpoint(checkpoint_path)

    model.test_model()


if __name__ == "__main__":
    main()
