
from src.pipelines.rnn import (
    get_env,
    get_model,
    get_data_generator,
)
import src.utils.torch as tu

import time

import numpy as np
import cv2

win_name = 'window'


def main():
    env = get_env()
    data_generator = get_data_generator(
        env,
        agent=lambda _: 2,
    )
    X, frames = next(data_generator)

    frames = frames.reshape(-1, *frames.shape[-3:])
    frames = np.transpose(frames, (0, 2, 3, 1)) / 255

    model = get_model(env)
    model.preload_weights()

    pred_frames = model(X)
    pred_frames = pred_frames.reshape(-1, *pred_frames.shape[-3:])
    pred_frames = pred_frames.permute(0, 2, 3, 1)
    pred_frames = pred_frames.detach().cpu().numpy()

    diff = abs(frames - pred_frames)
    grid = np.concatenate((frames, pred_frames, diff), axis=2)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 300)

    for frame in grid:
        time.sleep(1 / 10)
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
