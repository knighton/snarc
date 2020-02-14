import numpy as np

from .expert import Expert


def is_flipped(xx, yy_true, axis):
    for x, y_true in zip(xx, yy_true):
        y_pred = np.flip(x, axis)
        if not (y_pred == y_true).all():
            return False
    return True


class Flip(Expert):
    """
    Try flipping the board.
    """

    def fit(self, xx, yy):
        for x, y in zip(xx, yy):
            if x.shape != y.shape:
                return False
        for axis in [0, 1]:
            if is_flipped(xx, yy, axis):
                self.axis = axis
                return True
        return False

    def predict(self, x):
        return np.flip(x, self.axis)
