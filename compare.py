#! /usr/bin/env python3

import sys
import math

def get_scores(filename):
    scores = {}
    for line in open(filename):
        s = line.split()
        seed = int(s[0])
        score = int(s[1])
        scores[seed] = score
    return scores

a, b = sys.argv[1], sys.argv[2]
p, q = get_scores(a), get_scores(b)
seeds = list(set(p) & set(q))
sum_ratio = 0
win_p = 0
lose_p = 0
thresh = 0.1
for seed in seeds:
    x, y = p[seed], q[seed]
    if abs(x - y) / (x + 1e-10) > thresh:
        if x < y:
            win_p += 1
        elif x > y:
            lose_p += 1

    ratio = y / (x + 1e-10)
    if ratio < 20:
        sum_ratio += ratio

    if abs(x - y) / (x + 1e-10) > thresh:
        print('{:>5} {:>16} {:>16} {:>7.3f}'.format(seed, x, y, ratio))

total_ratio = sum_ratio / len(seeds)
win_p /= len(seeds)
lose_p /= len(seeds)
print('total_ratio: {}'.format(total_ratio))
print('win_p: {}'.format(win_p))
print('lose_p: {}'.format(lose_p))
