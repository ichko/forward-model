import os
import sys
import pickle

import numpy as np
import gym
import cv2

IS_DEBUG = '--debug' in sys.argv


class PreprocessedEnv:
    def __init__(
        self,
        env,
        obs_scalar=255,
        channel_first=True,
        use_rendering=False,
        shape=None,
    ):
        self.env = env
        self.use_rendering = use_rendering
        self.shape = shape
        self.channel_first = channel_first
        self.obs_scalar = obs_scalar

    def preprocess_obs(self, obs):
        if self.use_rendering:
            obs = self.env.render('rgb_array')
        if self.channel_first:
            obs = np.transpose(obs, (2, 0, 1))
        if self.shape is not None:
            obs = cv2.reshape(obs, self.shape)

        return obs / self.obs_scalar

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.preprocess_obs(self.env.reset())

    def step(self, *args):
        obs, reward, done, info = self.env.step(*args)
        return self.preprocess_obs(obs), reward, done, info


def make_preprocessed_env(env_name, *args, **kwargs):
    env = gym.make(env_name)
    return PreprocessedEnv(env, *args, **kwargs)


def persist(func, path, override=False):
    if not override and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    result = func()
    with open(path, 'wb+') as f:
        pickle.dump(result, f)

    return result


def add_virtual_display():
    # Necessary to display cartpole and other envs headlessly
    # https://stackoverflow.com/a/47968021
    from pyvirtualdisplay.smartdisplay import SmartDisplay

    display = SmartDisplay(visible=0, size=(1400, 900))
    display.start()

    # print(os.environ['DISPLAY'])


def try_colored_traceback():
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass


try_colored_traceback()
