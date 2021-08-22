import csv
from tqdm import tqdm
import pickle
import numpy as np
def obtain_subid2series():
    hid2time = {}
    admission_csv = open('mimicdata/mimic3/ADMISSIONS.csv')
    admission_csv = csv.reader(admission_csv)
    next(admission_csv)
    for row in admission_csv:
        hid = row[2]
        time = row[3]
        hid2time[hid] = time

    codes_csv = open('mimicdata/mimic3/ALL_CODES_filtered.csv')
    codes_csv = csv.reader(codes_csv)
    next(codes_csv)
    subid2series = {}
    for row in tqdm(codes_csv):
        sid = row[0]
        hid = row[1]
        time = hid2time[hid]
        code = row[2]
        if sid not in subid2series.keys():
            subid2series[sid] = {}

        if hid in subid2series[sid].keys():
            subid2series[sid][hid]['ICD_9CODE'].append(code)
        else:
            subid2series[sid][hid] = {'ADMITTIME': time, 'ICD_9CODE': [code], 'hid': hid}
    return subid2series, hid2time

def obtain_sequence_information(SUBJECT_ID,HADM_ID,ADMITTIME):
    records = subid2series[SUBJECT_ID]
    records = records.values()
    result = []
    for record in records:
        if record['ADMITTIME'] < ADMITTIME:
            result.append(record)
    return result

subid2series, hid2time = obtain_subid2series()

data_csv = open('mimicdata/mimic3/train_full.csv')
data_csv = csv.reader(data_csv)
next(data_csv)
lens = []
for key in subid2series.keys():
    lens.append(len(subid2series[key]))
lens = np.array(lens)
print(lens.mean())
for length in [1, 2, 3, 5, 10]:
    print(length)
    print(np.sum(lens>=length))

len_limits = [1, 2, 3, 5, 10]
len_benefits = {1:0, 2:0, 3:0, 5:0, 10:0}
for length_one in lens:
    for len_limit in len_limits:
        d = (length_one-len_limit)+1
        if d > 0:
            len_benefits[len_limit] += d
print(len_benefits)
print(np.sum(lens))
new_data = []
for row in data_csv:
    sid = row[0]
    hid = row[1]
    time = hid2time[hid]
    result = obtain_sequence_information(sid, hid, time)
    row.append(result)
    new_data.append(row)

index_list = [[],[],[],[],[]]
hid2pcodes = {}
hid2hids = {}
for key in subid2series.keys():
    cu_patient = subid2series[key]
    records = cu_patient.values()
    records = list(records)
    records = sorted(records, key=lambda s: s['ADMITTIME'])
    for tid, record in enumerate(records):
        hid2pcodes[int(record['hid'])] = []
        hid2hids[int(record['hid'])] = []
        if tid >= 0:
            for thre in range(1):
                index_list[thre].append(int(record['hid']))
        if tid >= 1:
            for thre in range(1,2):
                index_list[thre].append(int(record['hid']))
        if tid >= 2:
            for thre in range(2,3):
                index_list[thre].append(int(record['hid']))
        if tid >= 4:
            for thre in range(3,4):
                index_list[thre].append(int(record['hid']))
        if tid >= 9:
            for thre in range(4,5):
                index_list[thre].append(int(record['hid']))
        for pid in range(0, tid):
            hid2pcodes[int(record['hid'])].append(records[pid]['ICD_9CODE'])
        for pid in range(0, tid):
            hid2hids[int(record['hid'])].append(int(records[pid]['hid']))

pickle.dump(index_list, open('mimicdata/mimic3/index_list.pkl', 'wb'))
pickle.dump(hid2pcodes, open('mimicdata/mimic3/hid2pcodes.pkl', 'wb'))
pickle.dump(hid2hids, open('mimicdata/mimic3/hid2hids.pkl', 'wb'))
test_set = set(open('mimicdata/mimic3/test_full_hadm_ids.csv', 'r').read().splitlines())
index_list_test = [[],[],[],[],[]]
for k in range(5):
    indexs = index_list[k]
    for index in indexs:
        if str(index) in test_set:
            index_list_test[k].append(index)

pickle.dump(index_list_test, open('mimicdata/mimic3/index_list_test.pkl', 'wb'))