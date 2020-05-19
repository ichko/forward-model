
from src.pipelines.frame_transform import (
    get_env,
    get_model,
    hparams,
)
import src.utils.torch as tu

import time

import numpy as np
import cv2

win_name = 'window'

env = get_env()

def main():
    model = get_model(env)
    model.preload_weights()
    model = model.eval()

    true_frames = []
    actions = []
    done = False
    env.reset()

    while not done:
        frame = env.render('rgb_array')
        frame = cv2.resize(frame, hparams.frame_size)
        true_frames.append(frame)

        action = env.action_space.sample()
        # action = 2
        _, _, done, _, = env.step(action)
        actions.append(action)

    true_frames = np.array(true_frames)
    true_frames = np.transpose(true_frames, (0, 3, 1, 2))
    actions = np.array(actions)

    pred_frames = model.rollout(true_frames[:2], actions)

    true_frames = true_frames / 255.0
    diff = abs(true_frames - pred_frames)
    grid = np.concatenate((true_frames, pred_frames, diff), axis=3)
    grid = np.transpose(grid, (0, 2, 3, 1))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 300)

    for frame in grid:
        time.sleep(1 / 15)
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    while True:
        main()
