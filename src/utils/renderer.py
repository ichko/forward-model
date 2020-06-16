import sys
import numpy as np
import cv2

WIN_NAME = 'WINDOW'


def is_pressed(key):
    special = {
        32: 'space',
        81: 'left',
        82: 'up',
        83: 'right',
        84: 'down',
    }

    k = cv2.waitKey(33)
    char = special[k] if k in special else chr(k) if k > 0 else -1

    return char == key


class Renderer:
    WHITE = (1, 1, 1)
    RED = (0, 0, 1)

    def __init__(self, w, h):
        self.canvas = np.ones((w, h, 3))

        self.width = w
        self.height = h

        self.origin_x = self.width / 2
        self.origin_y = self.height / 2
        self.f = 0

    def clear(self):
        self.canvas = np.zeros((self.width, self.height, 3))

    def rect(self, x, y, w, h, rgb=(1, 1, 1), thickness=-1):
        x, y = self._origin_translate(x, y)
        cv2.rectangle(
            self.canvas,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            rgb,
            thickness=thickness,
        )

    def arc(self, x, y, rad, rgb=(1, 1, 1), thickness=-1):
        x, y = self._origin_translate(x, y)
        cv2.circle(
            self.canvas,
            (int(x), int(y)),
            int(rad),
            rgb,
            thickness=thickness,
        )

    def _origin_translate(self, x, y):
        return self.origin_x + x, self.origin_y - y

    @staticmethod
    def init_window(W=800, H=800):
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, W, H)
        cv2.moveWindow(WIN_NAME, 100, 100)

    @classmethod
    def can_render(cls):
        end = is_pressed('q')

        if end:
            cv2.destroyWindow(WIN_NAME)

        return not end

    @staticmethod
    def show_frame(canvas):
        cv2.waitKey(33)
        cv2.imshow(WIN_NAME, canvas)
