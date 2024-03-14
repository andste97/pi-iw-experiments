from pathlib import Path

import hydra
import wandb
from pytorch_lightning.loggers import WandbLogger

from models.lightning_model_basic import LightningDQN
from models.lightning_model_dynamic import LightningDQNDynamic
import timeit
from datetime import datetime
import gym
import gridenvs.examples  # register GE environments to gym

import pytorch_lightning as pl

@hydra.main(
    config_path="models/config",
    config_name="config_dynamic_explanatory_test.yaml",
    version_base="1.3",
)
def main(config):
    # set seeds, numpy for planner, torch for policy
    pl.seed_everything(config.train.seed)

    run = wandb.init(
        project="pi-iw-experiments-piiw",
        id=f'TEST_{config.train.env_id.replace("ALE/", "")}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")}',
        #offline=True,
        #log_model=False # needs to be False when offline is enabled
    )

    artifact = run.use_artifact('piiw-thesis/delayed_experience_replay/model-Riverraid-v4_2024-03-14_12-01-53.719976:v4', type='model')
    artifact_dir = artifact.download()
    checkpoint_path = Path(artifact_dir) / "model.ckpt"

    # choose wither to uses dynamic or BASIC features
    if config.model.use_dynamic_features:
        model = LightningDQNDynamic.load_from_checkpoint(checkpoint_path)
    else:
        model = LightningDQN.load_from_checkpoint(checkpoint_path)

    #trainer = pl.Trainer(
    #    accelerator="auto",
    #    max_epochs=config.train.max_epochs,
    #    logger=logger,
    #    deterministic="warn",
    #    enable_checkpointing=True
    #)

    #trainer.fit(
    #    model
    #)
    # save logged data
    #logger.save()

    model.test_model()


if __name__ == "__main__":
    main()
