from argparse import ArgumentParser
from glob import glob
import json
import numpy as np

from .solver import Solver


def parse_flags():
    x = ArgumentParser()
    x.add_argument('--in', type=str, default='data/training/')
    return x.parse_args()


def load_split(aa):
    xx = []
    yy = []
    for a in aa:
        x = a['input']
        y = a['output']
        x = np.array(x, np.uint8)
        y = np.array(y, np.uint8)
        xx.append(x)
        yy.append(y)
    return xx, yy


def load_task(f):
    x = json.load(open(f))
    t = load_split(x['train'])
    v = load_split(x['test'])
    return t, v


def is_correct(a, b):
    if a.shape != b.shape:
        return False
    return (a == b).all()


def score_task(true, pred):
    r = 0
    for t, p in zip(true, pred):
        r += is_correct(t, p)
    return r, len(true)


def run_task(solver, f):
    (tx, ty), (vx, vy_true) = load_task(f)
    vy_pred = solver.do_task(tx, ty, vx)
    return score_task(vy_true, vy_pred)


def main(flags):
    solver = Solver()
    d = getattr(flags, 'in')
    ff = glob(d + '/*.json')
    ff.sort()
    for i, f in enumerate(ff):
        r, z = run_task(solver, f)
        print(i, f, r, '/', z)


if __name__ == '__main__':
    main(parse_flags())
