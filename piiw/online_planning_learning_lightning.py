import hydra
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


@hydra.main(
    config_path="models/config",
    config_name="config_dynamic.yaml",
    version_base="1.3",
)
def main(config):
    # set seeds, numpy for planner, torch for policy
    pl.seed_everything(config.train.seed)

    if (not OmegaConf.is_config(config)):
        config = OmegaConf.create(config)

    logger = WandbLogger(
        project="delayed_experience_replay",
        id=f'{config.train.env_id.replace("ALE/", "")}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")}',
        #offline=True,
        #log_model=False # needs to be False when offline is enabled
        log_model='all'
    )


    # choose wither to uses dynamic or BASIC features
    if config.model.use_dynamic_features:
        model = LightningDQNDynamic(config)
    else:
        model = LightningDQN(config)

    logger.watch(model)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        #monitor="train/episode_reward",
        save_on_train_epoch_end=True,
        every_n_epochs=5
    )

    trainer = pl.Trainer(
        accelerator="auto",
        # max_epochs=config.train.max_epochs, # deprecated, use steps instead
        max_steps=config.train.max_steps,
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic="warn",
        enable_checkpointing=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
        num_sanity_val_steps=0
    )

    trainer.fit(
        model
    )
    # save logged data
    logger.save()

    model.validation_step({}, 1)


if __name__ == "__main__":
    main()
