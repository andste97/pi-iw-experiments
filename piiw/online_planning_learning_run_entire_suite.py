import multiprocessing
import random

import hydra
import numpy as np
import torch

import wandb

from models.model_dynamic import DQNDynamic
from datetime import datetime
import time
import gym
import gridenvs.examples  # register GE environments to gym

import pytorch_lightning as pl


@hydra.main(
    config_path="models/config",
    config_name="config_atari_dynamic.yaml",
    version_base="1.3",
)
def main(config):
    seeds = [0, 12345, 672354]
    processes = []
    process_info = {}

    for env_id in config.env_suite:
        group_name = f'group_{env_id}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")}'
        for seed in seeds:
            process = multiprocessing.Process(target=start_run, args=(env_id, seed, config, group_name))
            processes.append(process)
            process_info[process.name] = {"env": env_id, "seed": seed, "group": group_name, "start_time": time.time()}

    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

        end_time = time.time()
        runtime = end_time - process_info[process.name]["start_time"]
        exit_code = process.exitcode
        env = process_info[process.name]["env"]
        seed = process_info[process.name]["seed"]
        group_name = process_info[process.name]["group"]

        print(f"Process for env: {env}, seed: {seed} in group: '{group_name}' finished.")
        print(f"Runtime: {runtime:.2f} seconds, Exit Code: {exit_code}")

def start_run(env_id, seed, config, group_name):
    config.train.seed = seed
    config.train.env_id = env_id

    # set seeds, numpy for planner, torch for policy
    seed_everything(config.train.seed)

    # choose wither to uses dynamic or BASIC features
    model = DQNDynamic(config, group_name)
    model.fit(save_checkpoint_every_n_episodes=500)

    model.test_model()

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    main()
