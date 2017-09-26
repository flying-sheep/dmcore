#!/usr/bin/env python

with open('coords.txt') as c_f:
	d = [(int(i), int(j), float(v)) for i, j, v in (l.strip().split('\t') for l in c_f)]

fwd = set((a, b, v) for a, b, v in d if a < b)
bwd = set((a, b, v) for a, b, v in d if a > b)

if fwd != bwd:
	for l, r in zip(sorted(fwd - bwd), sorted(bwd - fwd)):
		print(l, r)
