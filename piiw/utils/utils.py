import numpy as np
import logging
import cv2
from distutils.util import strtobool
import argparse

logger = logging.getLogger(__name__)


def sample_cdf(cum_probs, size=None):
    s = cum_probs[-1]
    assert s > 0.99999 and s < 1.00001, f"Probabilities do not sum to 1: {cum_probs}" #just to check our input looks like a probability distribution, not 100% sure though.
    if size is None:
        # if rand >=s, cumprobs > rand would evaluate to all False. In that case, argmax would take the first element argmax([False, False, False]) -> 0.
        # This may happen still if probabilities sum to 1:
        # cumsum > rand is computed in a vectorized way, and in our machine (looks like) these operations are done in 32 bits.
        # Thus, even if our probabilities sum to exactly 1.0 (e.g. [0. 0.00107508 0.00107508 0.0010773 0.2831216 1.]), when rand is really close to 1 (e.g. 0.999999972117424),
        # when computing cumsum > rand in a vectorized way it will consider it in float32, which turns out to be cumsum > 1.0 -> all False.
        # This is why we check that (float32)rand < s:
        while True:
            rand = np.float32(np.random.rand())
            if rand < s:
                break
        res = (cum_probs > rand)
        return res.argmax()

    if type(size) is int:
        rand = np.random.rand(size).reshape((size,1))
    else:
        assert type(size) in (tuple, list), "Size can either be None for scalars, an int for vectors or a tuple/list containing the size for each dimension."
        assert len(size) > 0, "Use None for scalars."
        rand = np.random.rand(*size).reshape(size+(1,))
    # Again, we check that (float32)rand < s (easier to implement)
    mask = rand.astype(np.float32) >= s
    n = len(rand[mask])
    while n > 0:
        rand[mask] = np.random.rand(n)
        mask = rand.astype(np.float32) >= s
        n = len(rand[mask])
    return (cum_probs > rand).argmax(axis=-1)


def sample_pmf(probs, size=None):
    return sample_cdf(probs.cumsum(), size)


def random_index(array_len, size=None, replace=False, probs=None, cumprobs=None):
    """
    Similar to np.random.choice, but slightly faster.
    """
    if probs is None and cumprobs is None:
        res = np.random.randint(0, array_len, size)
        one_sample = lambda: np.random.randint(0, array_len)
    else:
        assert probs is None or cumprobs is None, "Either both probs and cumprobs is None (uniform probability distribution used) or only one of them is not None, not both."
        if cumprobs is None:
            cumprobs = probs.cumsum()
        assert array_len == len(cumprobs)
        res = sample_cdf(cumprobs, size)
        one_sample = lambda: sample_cdf(cumprobs)

    if not replace and size is not None:
        assert size <= array_len, "The array has to be longer than 'size' when sampling without replacement."
        s = set()
        for i in range(size):
            l = len(s)
            s.add(res[i])
            while len(s) == l:
                res[i] = one_sample()
                s.add(res[i])
    return res


def softmax(x, temp=1, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    if temp == 0:
        res = (x == np.max(x, axis=-1))
        return res/np.sum(res, axis=-1)
    x = x/temp
    e_x = np.exp( (x - np.max(x, axis=axis, keepdims=True)) ) #subtracting the max makes it more numerically stable, see http://cs231n.github.io/linear-classify/#softmax and https://stackoverflow.com/a/38250088/4121803
    return e_x / e_x.sum(axis=axis, keepdims=True)


def env_has_wrapper(env, wrapper_type):
    while env is not env.unwrapped:
        if isinstance(env, wrapper_type):
            return True
        env = env.env
    return False


def remove_env_wrapper(env, wrapper_type):
    if env is not env.unwrapped:
        if isinstance(env, wrapper_type):
            env = remove_env_wrapper(env.env, wrapper_type)
        else:
            env.env = remove_env_wrapper(env.env, wrapper_type)
    return env


def display_image_cv2(window_name, image, block_ms=1, size=None, image_filename=None):
    """
    Displays the given image with OpenCV2 in a window with the given name. It may also block until the window is close
    if block_ms is None, or for the given milliseconds. If 0 is given, it will actually block for 1ms, which is the
    minimum. If the data type of the image is integer, we assume it takes values in the range of integers [0,255]. If it
    is float, we assume it takes values in the range of real numbers [0,1].
    """
    if block_ms == 0: block_ms = 1 # it actually doesn't allow 0 ms
    elif block_ms is None: block_ms = 0 # 0 means until we close the window (None for us)
    assert block_ms >= 0
    if issubclass(image.dtype.type, np.integer): image = image.astype(np.float32)/255

    if(image.shape[0] == 3): # PIL-style RGB image with channels at beginning. Need to convert to opencv format
        image = np.transpose(image, (1, 2, 0))
    if size is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)

    if image_filename is not None:
        cv2.imwrite(filename=image_filename, img=image*255)

    cv2.imshow(window_name, image) # cv2 works with BGR (and also displays it like that)
    cv2.waitKey(block_ms) # shows image and waits for this amout of ms (or until we close the window if 0 is passed)

def transform_obs_to_image(obs):
    """This function transforms an observation consisting of 4 atari images into a single
    4x4 image. Upper-left side will be the 1st image, upper right 2nd, lower left 3rd,
    and lower right 4th.
    Each single image needs to be a grayscale numpy array of shape: (width, height), values uint8."""

    # Horizontal stacking
    top_half = np.hstack([obs[0], obs[1]])  # Combine a1 (upper left) and a2 (upper right)
    bottom_half = np.hstack([obs[2], obs[3]])  # Combine a3 (lower left) and a4 (lower right)

    # Vertical stacking
    return np.vstack([top_half, bottom_half])


class AnsiSpecial:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    colors = {
       "purple" : '\033[95m',
       "cyan" : '\033[96m',
       "darkcyan" : '\033[36m',
       "blue" : '\033[94m',
       "green" : '\033[92m',
       "yellow" : '\033[93m',
       "red" : '\033[91m',
    }


def cstr(s, color=None, bold=False, underline=False):
    assert color is not None or bold or underline
    s = str(s)
    if color is not None:
        assert color in AnsiSpecial.colors.keys(), f"Color not in {list(AnsiSpecial.colors.keys())}"
        header = AnsiSpecial.colors[color]
    else:
        header = ""
    if bold:
        header += AnsiSpecial.BOLD
    if underline:
        header += AnsiSpecial.UNDERLINE

    s = s.replace(AnsiSpecial.END, AnsiSpecial.END+header)

    if not s.endswith(AnsiSpecial.END):
        s += '\033[0m'
    return header + s

def reward_in_tree(tree):
    iterator = iter(tree)
    next(iterator)  # discard root
    for node in iterator:
        if node.data["r"] > 0:
            return True
    return False
