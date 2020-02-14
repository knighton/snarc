from argparse import ArgumentParser
from glob import glob


def parse_flags():
    x = ArgumentParser()
    x.add_argument('--in', type=str, default='data/training/')
    return x.parse_args()


def main(flags):
    d = getattr(flags, 'in')
    ff = glob(d + '/*.json')
    ff.sort()
    for i, f in enumerate(ff):
        print(i, f)


if __name__ == '__main__':
    main(parse_flags())
