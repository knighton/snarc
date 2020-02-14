from argparse import ArgumentParser
from glob import glob
import json
import numpy as np

from .solver import Solver


def parse_flags():
    x = ArgumentParser()
    x.add_argument('--in', type=str, default='data/training/')
    return x.parse_args()


def load_split(xx):
    pairs = []
    for x in xx:
        i = x['input']
        o = x['output']
        i = np.array(i, np.uint8)
        o = np.array(o, np.uint8)
        pairs.append((i, o))
    return pairs


def load_task(f):
    x = json.load(open(f))
    trains = load_split(x['train'])
    tests = load_split(x['test'])
    return trains, tests


def run_task(solver, f):
    trains, tests = load_task(f)
    # TODO


def main(flags):
    solver = Solver()
    d = getattr(flags, 'in')
    ff = glob(d + '/*.json')
    ff.sort()
    for i, f in enumerate(ff):
        run_task(solver, f)


if __name__ == '__main__':
    main(parse_flags())
