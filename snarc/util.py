import numpy as np


def get_background_color(x):
    x = np.bincount(x.flatten())
    return x.argmax()
