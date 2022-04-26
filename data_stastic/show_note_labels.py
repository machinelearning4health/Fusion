import csv
import re
from constants import *
from nltk import pos_tag
import nltk
from tqdm import tqdm
import numpy as np
pos= ['NN','NNS','JJ']
notes_label_path = '%s/notes_labeled_original.csv' % MIMIC_3_DIR
ICD9_dict = {}
notes_label_file = open(notes_label_path, 'r', encoding='utf-8')

ICD9_des_path = '%s/D_ICD_DIAGNOSES.csv' % DATA_DIR
ICD9_description = csv.reader(open(ICD9_des_path, 'r', encoding='utf-8'))
next(ICD9_description)
for description in ICD9_description:
    code = description[1]
    short_name = description[2]
    long_name = description[3]
    ICD9_dict[code] = (short_name.lower(), long_name.lower())
ICD9_des_path = '%s/D_ICD_PROCEDURES.csv' % DATA_DIR
ICD9_description = csv.reader(open(ICD9_des_path, 'r', encoding='utf-8'))
next(ICD9_description)
for description in ICD9_description:
    code = description[1]
    short_name = description[2]
    long_name = description[3]
    ICD9_dict[code] = (short_name.lower(), long_name.lower())

ICD9_list_path = '%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR
ICD9_list = csv.reader(open(ICD9_list_path, 'r', encoding='utf-8'))
ICD9_set = set()
next(ICD9_list)
for row in ICD9_list:
    code = row[2]
    code = re.sub('\.', '', code)
    ICD9_set.add(code)

csv_reader = csv.reader(notes_label_file)
next(csv_reader)
avg_list = []
ct = 0
for row in tqdm(csv_reader):
    ct += 1
    if ct > 10000:
        break
    text = row[2]
    codes = row[3].split(';')
    #print(text)
    for code in codes:
        count_short = 0
        count_long = 0
        code = re.sub('\.', '', code)
        if code not in ICD9_dict:
            continue
        temps_short = nltk.word_tokenize(ICD9_dict[code][0])
        rs_short = pos_tag(temps_short)
        new_temps = []
        for word, type in rs_short:
            if type in pos:
                new_temps.append(word)
        temps_short = new_temps
        for temp in temps_short:
            if temp.lower() in text.lower():
                count_short += 1

        temps_long = nltk.word_tokenize(ICD9_dict[code][1])
        rs_long = pos_tag(temps_long)
        new_temps = []
        for word, type in rs_long:
            if type in pos:
                new_temps.append(word)
        temps_long = new_temps
        for temp in temps_long:
            if temp.lower() in text.lower():
                count_long += 1
        avg_list.append([count_short/(len(temps_short)+1e-10), count_long/(len(temps_long)+1e-10)])
        #print(ICD9_dict[code], count_short/(len(temps_short)+1e-10), count_long/(len(temps_long)+1e-10))
avg_list = np.array(avg_list)
print(avg_list.mean(axis=0))