import csv
from constants import *
base_name = "%s/disch" % MIMIC_3_DIR
train_name = '%s_train_split_tf_general.csv' % (base_name)
train_file = open(train_name, 'r')
reader = csv.reader(train_file)
next(reader)
count = 0
for row in reader:
    if count > 10:
        break
    count += 1
    print(row[1])
    tokens = row[2].split()
    tokens = tokens[0:2500]
    entitys = row[4].split()
    entitys = entitys[0:2500]
    negations = row[5].split()
    negations = negations[0:2500]
    for tid, (token, entity, negation) in enumerate(zip(tokens, entitys, negations)):
        if entity == '#1#':
            token = '\033[27;32;40m['+token+']\033[0m'
        if negation == '#1#':
            token = '\033[27;31;40m|'+token+'|\033[0m'
        tokens[tid] = token
    txt = ' '.join(tokens)
    print(txt)
