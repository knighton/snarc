from .expert.flip import Flip
from .expert.rotate import Rotate


class Solver(object):
    def __init__(self):
        self.experts = [Flip(), Rotate()]

    def do_task(self, train_xx, train_yy, test_xx):
        for expert in self.experts:
            ok = expert.fit(train_xx, train_yy)
            if not ok:
                continue
            return [expert.predict(x) for x in test_xx]
        return test_xx
