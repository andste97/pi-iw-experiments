from omegaconf import DictConfig
import torch
import pytorch_lightning as pl


def configure_optimizer_based_on_config(model: torch.nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    if "train.optim" in config:
        if config.train.optim == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.train.learning_rate,
                eps=config.train.rmsprop_epsilon,
                weight_decay=config.train.l2_reg_factor
            )
        elif config.train.optim == "adam":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.train.learning_rate,
                eps=config.train.rmsprop_epsilon,
                weight_decay=config.train.l2_reg_factor
            )
        else:
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=config.train.learning_rate,
                alpha=config.train.rmsprop_alpha,  # same as rho in tf
                eps=config.train.rmsprop_epsilon,
                weight_decay=config.train.l2_reg_factor
                # use this for l2 regularization to replace TF regularization implementation
            )
    else:
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.train.learning_rate,
            alpha=config.train.rmsprop_alpha,  # same as rho in tf
            eps=config.train.rmsprop_epsilon,
            weight_decay=config.train.l2_reg_factor
            # use this for l2 regularization to replace TF regularization implementation
        )

    return optimizer
