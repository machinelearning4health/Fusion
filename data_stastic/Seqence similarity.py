import csv
from tqdm import tqdm
import pickle
import numpy as np
def obtain_subid2series():
    hid2time = {}
    admission_csv = open('../mimicdata/mimic3/ADMISSIONS.csv')
    admission_csv = csv.reader(admission_csv)
    next(admission_csv)
    for row in admission_csv:
        hid = row[2]
        time = row[3]
        hid2time[hid] = time

    codes_csv = open('../mimicdata/mimic3/ALL_CODES_filtered.csv')
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
import time
subid2series, hid2time = obtain_subid2series()
similarity_list=[[] for k in range(10)]
similarity_time_list=[[] for k in range(6)]
for key in subid2series.keys():
    cu_patient = subid2series[key]
    records = cu_patient.values()
    records = list(records)
    records = sorted(records, key=lambda s: s['ADMITTIME'])
    for tid, record in enumerate(records):
        for k in range(0, min(tid, 10)):
            a = set(records[tid]['ICD_9CODE'])
            b = set(records[tid - k - 1]['ICD_9CODE'])
            simi = len(a.intersection(b)) / len(b)
            similarity_list[k].append(simi)
            time_a = int(time.mktime(time.strptime(records[tid]['ADMITTIME'], "%Y-%m-%d %H:%M:%S")))
            time_b = int(time.mktime(time.strptime(records[tid-k-1]['ADMITTIME'], "%Y-%m-%d %H:%M:%S")))
            d = (time_a-time_b)/(60*60*24)
            if d<=7:
                similarity_time_list[0].append(simi)
            elif d<=30:
                similarity_time_list[1].append(simi)
            elif d<=180:
                similarity_time_list[2].append(simi)
            elif d<=360:
                similarity_time_list[3].append(simi)
            elif d<=720:
                similarity_time_list[4].append(simi)
            else:
                similarity_time_list[5].append(simi)

for k in range(10):
    similarity_list[k] = np.array(similarity_list[k])
    print(len(similarity_list[k]))
    print(similarity_list[k].mean())
    print(similarity_list[k].std())
    print('______________________________________________')
print("based on TIME")
for k in range(6):
    similarity_time_list[k] = np.array(similarity_time_list[k])
    print(len(similarity_time_list[k]))
    print(similarity_time_list[k].mean())
    print(similarity_time_list[k].std())
    print('______________________________________________')
