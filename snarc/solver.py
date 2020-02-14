from .expert.fall import Fall
from .expert.flip import Flip
from .expert.rotate import Rotate
from .expert.zoom_on_frame import ZoomOnFrame
from .expert.zoom_on_object import ZoomOnObject


class Solver(object):
    def __init__(self):
        self.experts = [
            Fall(),
            Flip(),
            Rotate(),
            ZoomOnFrame(),
            ZoomOnObject(),
        ]

    def do_task(self, train_xx, train_yy, test_xx):
        for expert in self.experts:
            if not expert.fit(train_xx, train_yy):
                continue
            return [expert.predict(x) for x in test_xx]
        return test_xx
