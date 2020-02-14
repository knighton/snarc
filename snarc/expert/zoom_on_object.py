import numpy as np

from ..util import get_background_color
from .expert import Expert


def zoom_on_object(a):
    bg_color = get_background_color(a)
    y, x = np.where(a != bg_color)
    if not len(y):
        return None
    y.sort()
    x.sort()
    return a[y[0]:y[-1] + 1, x[0]:x[-1] + 1]


class ZoomOnObject(Expert):
    def fit(self, xx, yy):
        for x, y_true in zip(xx, yy):
            y_pred = zoom_on_object(x)
            if not np.array_equal(y_true, y_pred):
                return False
        return True

    def predict(self, x):
        return zoom_on_object(x)
