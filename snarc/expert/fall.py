import numpy as np

from ..util import get_background_color
from .expert import Expert


def get(a, bg_color):
    ccc = []
    for x in range(a.shape[1]):
        cc = []
        for y in range(a.shape[0]):
            c = a[y, x]
            if c != bg_color:
                cc.append(c)
        ccc.append(reversed(cc))
    return ccc


def put(in_shape, ccc, bg_color):
    out = np.zeros(in_shape, np.int32)
    for i, cc in enumerate(ccc):
        for j, c in enumerate(cc):
            out[-j - 1, i] = c
    return out


def fall(a):
    bg_color = get_background_color(a)
    ccc = get(a, bg_color)
    return put(a.shape, ccc, bg_color)


class Fall(Expert):
    def fit(self, xx, yy):
        for x, y_true in zip(xx, yy):
            y_pred = fall(x)
            if not np.array_equal(y_true, y_pred):
                return False
        return True

    def predict(self, x):
        return fall(x)
