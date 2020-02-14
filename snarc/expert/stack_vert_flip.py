import numpy as np

from .expert import Expert


def stack_vert_flip(x):
    h = x.shape[0]
    if h % 2:
        return None
    a = x[h // 2:, ]
    a_flip = np.flip(a, 0)
    return np.concatenate([a_flip, a], 0)


class StackVertFlip(Expert):
    def fit(self, xx, yy):
        for x, y_true in zip(xx, yy):
            y_pred = stack_vert_flip(x)
            if not np.array_equal(y_true, y_pred):
                return False
        return True

    def predict(self, x):
        return stack_vert_flip(x)
