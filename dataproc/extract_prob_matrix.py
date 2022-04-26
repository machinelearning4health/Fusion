import pickle
import csv
import numpy as np
from utils import load_lookups_old
from options import args
filename = '../mimicdata/mimic3/train_full.csv'
dicts = load_lookups_old(args)
ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
num_labels = len(ind2c)
co_occur_matrix = np.zeros((num_labels, num_labels))
c_matrix = np.zeros((num_labels, 1))
with open(filename, 'r') as infile:
    r = csv.reader(infile)
    # header
    next(r)

    for row in r:

        text = row[2]
        codes = list(set(row[3].split(';')))
        for l1 in codes:
            if l1 in c2ind.keys():
                c_matrix[c2ind[l1], 0] += 1
            for l2 in codes:
                if l1 in c2ind.keys() and l2 in c2ind.keys():
                    code1 = int(c2ind[l1])
                    code2 = int(c2ind[l2])
                    co_occur_matrix[code1, code2] += 1
co_occur_matrix_prob = co_occur_matrix / (c_matrix+1e-3)
freq_prob = c_matrix[:, 0].astype(np.int)
PC = 1/len(freq_prob)/freq_prob
PC = np.clip(PC, 0, 1)
freq_prob = np.clip(freq_prob, 0, 9999999999999999)

max_freq = freq_prob.max()
count = np.zeros(max_freq+1, dtype=np.int)
for cid, freq in enumerate(freq_prob):
    count[freq] += 1

np.save('../mimicdata/mimic3/co-occurrence.npy', co_occur_matrix_prob)
np.save('../mimicdata/mimic3/a-freq.npy', freq_prob)
np.save('../mimicdata/mimic3/PC.npy', PC)
print(co_occur_matrix)
