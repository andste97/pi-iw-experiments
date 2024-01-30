import gym
import cv2
import numpy as np
from collections import deque


# Gym wrapper with clone/restore state
class Wrapper(gym.Wrapper):
    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        self.env.restore_state(state)


class ResizeImage(Wrapper):
    def __init__(self, env, new_size):
        super(ResizeImage, self).__init__(env)
        self.resize_fn = lambda obs: cv2.resize(obs, dsize=new_size, interpolation=cv2.INTER_LINEAR)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_size)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.resize_fn(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.resize_fn(observation), reward, done, info


class FrameBuffer(Wrapper):
    def __init__(self, env, buffer_size):
        assert (buffer_size > 0)
        super(FrameBuffer, self).__init__(env)
        self.buffer_size = buffer_size
        self.observations = deque(maxlen=buffer_size)
        shape = [self.buffer_size] + list(self.observation_space.shape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape)

    def step(self, a):
        observation, reward, done, info = self.env.step(a)
        self.observations.append(observation)
        return self.observation(), reward, done, info

    def reset(self):
        initial_frame = self.env.reset()
        for _ in range(self.buffer_size):
            self.observations.append(initial_frame)
        return self.observation()

    def observation(self):
        # Return a list instead of a numpy array to reduce space in memory when storing the same frame more than once
        return list(self.observations)

    def clone_state(self):
        return (tuple(self.observations), self.env.clone_state())

    def restore_state(self, state):
        assert len(state[0]) == len(self.observations)
        self.observations.extend(state[0])
        return self.env.restore_state(state[1])


class MaxSteps(Wrapper):
    def __init__(self, env, max_steps):
        assert (max_steps > 0)
        self.max_steps = max_steps
        self._steps = 0
        super(MaxSteps, self).__init__(env)

    def step(self, a):
        obs, r, done, info = self.env.step(a)
        self._steps += 1
        if self._steps >= self.max_steps:
            done = True
        return obs, r, done, info

    def reset(self):
        self._steps = 0
        return self.env.reset()

    def clone_state(self):
        return (self._steps, self.env.clone_state())

    def restore_state(self, state):
        self._steps = state[0]
        self.env.restore_state(state[1])

def downsample_goexplore(state, downsampling_tiles_w=8, downsampling_tiles_h=11, max_pix_value=7, grayscale=True):
    """
    Downsamples an RGB image to a given shape and converts it into grayscale as in Go-Explore
    https://github.com/uber-research/go-explore/blob/4dc469002769bdcb1115fbc2e0eafb20885dcf55/goexplore_py/montezuma_env.py#L47
    """
    shape = (downsampling_tiles_w, downsampling_tiles_h)
    if grayscale and len(state.shape) == 3:
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, shape, interpolation=cv2.INTER_AREA)
    if max_pix_value < 255:
        state = state/255.0 * max_pix_value
        state = state.astype(np.uint8)
    return state


class Downsampling(Wrapper):
    def __init__(self, env, downsampling_tiles_w=8, downsampling_tiles_h=11, downsampling_pixel_values=8):
        super(Downsampling, self).__init__(env)
        self.downsampling_tiles_w = downsampling_tiles_w
        self.downsampling_tiles_h = downsampling_tiles_h
        self.downsampling_max_pix_value = downsampling_pixel_values - 1

    def reset(self):
        obs = self.env.reset()
        self.unwrapped.downsampled_image = downsample_goexplore(obs,
                                                                downsampling_tiles_w=self.downsampling_tiles_w,
                                                                downsampling_tiles_h=self.downsampling_tiles_h,
                                                                max_pix_value=self.downsampling_max_pix_value)
        return obs

    def step(self, a):
        obs, r, done, info = self.env.step(a)
        self.unwrapped.downsampled_image = downsample_goexplore(obs,
                                                                downsampling_tiles_w=self.downsampling_tiles_w,
                                                                downsampling_tiles_h=self.downsampling_tiles_h,
                                                                max_pix_value=self.downsampling_max_pix_value)
        return obs, r, done, info


class FullCloneRestore(Wrapper):
    def clone_state(self):
        return self.env.clone_full_state()

    def restore_state(self, state):
        self.env.restore_full_state(state)


def is_atari_env(env):
    import gym.envs.atari
    return isinstance(env.unwrapped, gym.envs.atari.AtariEnv)


def wrap_atari_env(env, frameskip, max_steps, add_downsampling, downsampling_tiles_w, downsampling_tiles_h, downsampling_pix_values):
    # To get grayscale images, instead of wrapping the env, we modify the _get_obs function
    # this way, ale.getScreenGrayscale function is called instead of ale.getScreenRGB2
    # The RGB image will still show when rendering.
    screen_width, screen_height = env.unwrapped.ale.getScreenDims()
    env.unwrapped._get_obs = lambda : env.unwrapped.ale.getScreenGrayscale().reshape((screen_height, screen_width))
    env.unwrapped.frameskip = frameskip

    env = FullCloneRestore(env)

    if add_downsampling:
        env = Downsampling(env, downsampling_tiles_w=downsampling_tiles_w, downsampling_tiles_h=downsampling_tiles_h,
                           downsampling_pixel_values=downsampling_pix_values)
    env = ResizeImage(env, new_size=(84, 84))
    env = FrameBuffer(env, buffer_size=4)
    if max_steps is not None:
        env = MaxSteps(env, max_steps=max_steps)
    return env