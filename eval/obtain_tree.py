import pickle
import numpy as np
from sklearn.metrics import f1_score
from utils import macro_f1
def level_1_f1():
    yhat = np.zeros((len(tar_icd), 178))
    y = np.zeros((len(tar_icd), 178))
    for pre_group, tar_group in zip(pre_icd, tar_icd):
        for icd in pre_group:
            icd = icd.split('.')[0]
            if len(icd) < 3:
                icd = '0'*(3-len(icd))+icd
            if icd not in icd_diction:
                continue
            yhat[icd_diction[icd]] = 1
        for icd in tar_group:
            icd = icd.split('.')[0]
            if len(icd) < 3:
                icd = '0'*(3-len(icd))+icd
            if icd not in icd_diction:
                continue
            y[icd_diction[icd]] = 1
    macro_f1_score = macro_f1(yhat, y)
    print(macro_f1_score)

def level_2_f1():
    yhat = np.zeros((len(tar_icd), len(icdlv2_diction)))
    y = np.zeros((len(tar_icd), len(icdlv2_diction)))
    for pre_group, tar_group in zip(pre_icd, tar_icd):
        for icd in pre_group:
            icd = icd.split('.')[0]
            if len(icd) < 3:
                icd = '0'*(3-len(icd))+icd
            if icd not in icdlv2_diction:
                continue
            yhat[icdlv2_diction[icd]] = 1
        for icd in tar_group:
            icd = icd.split('.')[0]
            if len(icd) < 3:
                icd = '0'*(3-len(icd))+icd
            if icd not in icdlv2_diction:
                continue
            y[icdlv2_diction[icd]] = 1
    macro_f1_score = macro_f1(yhat, y)
    print(macro_f1_score)

def topk_hit_rate():
    for k in range(1, 10):
        score = 0
        for pre_group, tar_group in zip(pre_icd_top10, tar_icd):
            pre_group = pre_group[0:k]
            tar_group = set(tar_group)
            count = 0
            for one in pre_group:
                if one in tar_group:
                    count += 1
            score += count/len(pre_group)/len(tar_icd)

        print(score)
import copy
def topk_hit_rate_lv0():
    for k in range(1, 10):
        score = 0
        tar_icd_ = copy.deepcopy(tar_icd)
        for pre_group, tar_group in zip(pre_icd_top10, tar_icd_):
            pre_group = pre_group[0:k]
            for pid in range(len(pre_group)):
                icd = pre_group[pid]
                icd = icd.split('.')[0]
                if len(icd) < 3:
                    icd = '0' * (3 - len(icd)) + icd
                if icd not in icd_diction:
                    continue
                pre_group[pid] = icd_diction[icd]
            for tid in range(len(tar_group)):
                icd = tar_group[tid]
                icd = icd.split('.')[0]
                if len(icd) < 3:
                    icd = '0' * (3 - len(icd)) + icd
                if icd not in icd_diction:
                    continue
                tar_group[tid] = icd_diction[icd]
            tar_group = set(tar_group)
            count = 0
            for one in pre_group:
                if one in tar_group:
                    count += 1
            score += count/len(pre_group)/len(tar_icd)

        print(score)

def topk_hit_rate_lv1():
    for k in range(1, 10):
        score = 0
        tar_icd_ = copy.deepcopy(tar_icd)
        for pre_group, tar_group in zip(pre_icd_top10, tar_icd_):
            pre_group = pre_group[0:k]
            for pid in range(len(pre_group)):
                icd = pre_group[pid]
                icd = icd.split('.')[0]
                if len(icd) < 3:
                    icd = '0' * (3 - len(icd)) + icd
                if icd not in icdlv2_diction:
                    continue
                pre_group[pid] = icdlv2_diction[icd]
            for tid in range(len(tar_group)):
                icd = tar_group[tid]
                icd = icd.split('.')[0]
                if len(icd) < 3:
                    icd = '0' * (3 - len(icd)) + icd
                if icd not in icdlv2_diction:
                    continue
                tar_group[tid] = icdlv2_diction[icd]
            tar_group = set(tar_group)
            count = 0
            for one in pre_group:
                if one in tar_group:
                    count += 1
            score += count/len(pre_group)/len(tar_icd)

        print(score)

diction_icd = pickle.load(open('./icd9_category.pk', 'rb'))
icd_diction = {}
icdlv2_diction = {}
for kid, key in enumerate(diction_icd.keys()):
    for value in diction_icd[key]:
        icd_diction[value] = kid
        icdlv2_diction[value] = len(icdlv2_diction)
pre_icd = pickle.load(open('./pre_icd.pkl', 'rb'))
pre_icd_top10 = pickle.load(open('./pre_icd_top10.pkl', 'rb'))
tar_icd = pickle.load(open('./tar_icd.pkl', 'rb'))
topk_hit_rate()