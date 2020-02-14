import numpy as np

from .expert import Expert


def is_rotated(xx, yy_true, k):
    for x, y_true in zip(xx, yy_true):
        y_pred = np.rot90(x, k)
        if not (y_pred == y_true).all():
            return False
    return True


class Rotate(Expert):
    """
    Try rotating the board.
    """

    def fit(self, xx, yy):
        for x, y in zip(xx, yy):
            if x.shape != y.shape:
                return False
            if x.shape[0] != x.shape[1]:
                return False
        for k in [0, 1, 2]:
            if is_rotated(xx, yy, k):
                self.k = k
                return True
        return False

    def predict(self, x):
        return np.rot90(x, self.k)
