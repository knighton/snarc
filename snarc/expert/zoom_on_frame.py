import numpy as np

from .expert import Expert


def from_top_left(a, y0, x0):
    c = a[y0, x0]
    if a[y0 + 1, x0] != c:
        return None
    if a[y0 + 2, x0] != c:
        return None
    if a[y0, x0 + 1] != c:
        return None
    if a[y0, x0 + 2] != c:
        return None
    for y in range(y0 + 2, a.shape[0]):
        if a[y, x0] != c:
            break
    y1 = y - 1
    for x in range(x0 + 2, a.shape[1]):
        if a[y0, x] != c:
            break
    x1 = x - 1
    for y in range(y0, y1 + 1):
        if a[y, x1] != c:
            return None
    for x in range(x0, x1 + 1):
        if a[y1, x] != c:
            return None
    area = (y1 - y0) * (x1 - x0)
    coords = (y0, x0), (y1, x1)
    return area, coords


def get_diff_color_frame_among_noise(a):
    pairs = []
    for y in range(a.shape[0] - 2):
        for x in range(a.shape[1] - 2):
            r = from_top_left(a, y, x)
            if not r:
                continue
            area, coords = r
            pairs.append((area, coords))
    pairs.sort()
    return pairs[0][1] if pairs else None


def zoom_on_frame(a):
    z = get_diff_color_frame_among_noise(a)
    if z is None:
        return None
    (y0, x0), (y1, x1) = z
    return a[y0:y1 + 1, x0:x1 + 1]


class ZoomOnFrame(Expert):
    def fit(self, xx, yy):
        for x, y_true in zip(xx, yy):
            y_pred = zoom_on_frame(x)
            if not np.array_equal(y_true, y_pred):
                return False
        return True

    def predict(self, x):
        return zoom_on_frame(x)
