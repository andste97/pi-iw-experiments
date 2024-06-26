import gym
from utils.utils import env_has_wrapper, remove_env_wrapper
from atari_utils.atari_wrappers import is_atari_env, wrap_atari_env
import numpy as np
import logging as logger

def make_env(env_id: str, max_episode_steps: int, atari_frameskip: int):

    if(env_id.startswith('GE')):
        # Gridenv does not support frameskip parameter
        env = gym.make(env_id)
    else:
        # Create the gym environment for Atari.
        # When creating it with gym.make(), gym usually puts a TimeLimit wrapper around an env.
        # We'll take this out since we will most likely reach the step limit (when restoring the internal state of the
        # emulator the step count of the wrapper will not reset)
        env = gym.make(env_id, frameskip=atari_frameskip)


    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
        logger.info("TimeLimit environment wrapper removed.")

    from gridenvs.env import GridEnv
    if isinstance(env, GridEnv):
        env.max_moves = max_episode_steps

    # If the environment is an Atari game, the observations will be the last four frames stacked in a 4-channel image
    if is_atari_env(env):
        env = wrap_atari_env(env, max_steps=max_episode_steps)
        logger.info("Atari environment modified: observation is now a 4-channel image of the last four non-skipped frames in grayscale. Frameskip set to %i." % atari_frameskip)

    return env