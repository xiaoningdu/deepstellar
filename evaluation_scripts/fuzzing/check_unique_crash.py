import os
import numpy as np
import argparse
import xxhash


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control experiment')
    parser.add_argument('-i', help='crash dir path')
    args = parser.parse_args()

    dirs = os.listdir(args.i)
    hash_set = set()
    for i in dirs:
        crash_seed = os.path.join(args.i, i)
        seed = np.load(crash_seed)

        h = xxhash.xxh64()
        h.update(seed)
        q = h.intdigest()
        if q not in hash_set:
            hash_set.add(q)

    print(len(hash_set))
    print('finish')
