import os
import sys
import pickle

import numpy as np
import gym
import cv2

IS_DEBUG = '--debug' in sys.argv

_registered_envs = dict()
_old_gym_make = gym.make


def _new_gym_make(id, *args, **kwargs):
    if id in _registered_envs:
        return _registered_envs[id](*args, **kwargs)

    return _old_gym_make(id, *args, **kwargs)


# monkey patch gym.make
gym.make = _new_gym_make


def register_gym_env(id, cls):
    _registered_envs[id] = cls


class PreprocessedEnv:
    def __init__(
        self,
        env,
        obs_scalar=256,
        channel_first=True,
        use_rendering=True,
        frame_size=None,
    ):
        self.env = env
        self.use_rendering = use_rendering
        self.frame_size = frame_size
        self.channel_first = channel_first
        self.obs_scalar = obs_scalar

    def preprocess_obs(self, obs):
        if self.use_rendering:
            obs = self.env.render('rgb_array')
        if self.frame_size is not None:
            obs = cv2.resize(obs, self.frame_size)
        if self.channel_first:
            obs = np.transpose(obs, (2, 0, 1))

        return obs / self.obs_scalar

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.preprocess_obs(self.env.reset())

    def step(self, *args):
        obs, reward, done, info = self.env.step(*args)
        obs = self.preprocess_obs(obs)
        return obs, reward, done, info


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


def cartesian(*lists):
    result = [[]]
    for l in lists:
        result = [r + [el] for r in result for el in l]

    return result


# This is necessary for some openai gym environments rendering
def add_virtual_display_if_non_present():
    if 'DISPLAY' not in os.environ:
        # Necessary to display cartpole and other envs headlessly
        # https://stackoverflow.com/a/47968021
        from pyvirtualdisplay.smartdisplay import SmartDisplay

        display = SmartDisplay(visible=0, size=(1400, 900))
        display.start()
        display = os.environ['DISPLAY']

        print(f'>> ADDED VIRTUAL DISPLAY [{display}]')


add_virtual_display_if_non_present()


def try_colored_traceback():
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass


try_colored_traceback()


# This does `not` work with `down` because the variable is not shared between threads
# `d` variable is shared
# TODO: make variable shared?
def keyboard():
    from pynput import keyboard

    down = set()
    d = None

    class Keyboard:
        def is_pressed(self, key):
            if hasattr(d, 'char'):
                return key == d.char
            return False
            # return key in down

        def on_release(self, key):
            if key in down:
                down.remove(key)

        def on_press(self, key):
            nonlocal d
            d = key
            down.add(key)

        def __enter__(self):
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
            )
            self.listener.__enter__()

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.listener.__exit__(exc_type, exc_val, exc_tb)

    return Keyboard()


def numpy_img_dims(imgs):
    return np.transpose(imgs, (0, 2, 3, 1))


def get_example_rollout(info, id=0, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    if show:
        y = info['y'][id]
        y_pred = info['y_pred'][id]
    else:
        y = info['y'][id].detach().cpu().numpy()
        y_pred = info['y_pred'][id].detach().cpu().numpy()

    y = numpy_img_dims(y)
    y_pred = numpy_img_dims(y_pred)
    diff = abs(y - y_pred)

    num_imgs = 12
    img_range = list(enumerate(range(0, len(y), len(y) // num_imgs + 1)))
    num_imgs = len(img_range)

    plot_size = 2
    fig, axs = plt.subplots(
        3,
        num_imgs,
        figsize=(plot_size * num_imgs, plot_size * 3),
    )

    for i, f in img_range:
        l, r, m = (axs[0, i], axs[1, i], axs[2, i])

        l.imshow(y[f])
        r.imshow(y_pred[f])
        m.imshow(diff[f])

        l.set_xticklabels([])
        r.set_xticklabels([])
        m.set_xticklabels([])
        l.set_yticklabels([])
        r.set_yticklabels([])
        m.set_yticklabels([])

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if show:
        plt.show()
    else:
        canvas = FigureCanvas(fig)
        canvas.draw()  # https://stackoverflow.com/a/35362787

        buf = canvas.buffer_rgba()
        # convert to a NumPy array
        image = np.asarray(buf)
        plt.close()

    return image
