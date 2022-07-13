from multiprocessing import Pool
import pickle
import numpy as np
import math
import networkx as nx
import time
import numba
import sys
import random
import pandas as pd
import yaml
import os

config = yaml.safe_load(open('config.yaml'))
dataset = str(config["dataset"])
dataset_point = config["pointnum"][str(config["dataset"])]

def find_trajtimelist():
    longest_traj = 0
    smallest_time = np.inf
    time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
    for time_list in time_list_int:
        if len(time_list)>longest_traj:
            longest_traj = len(time_list)
        for t in time_list:
            if t<smallest_time:
                smallest_time = t
    return longest_traj, smallest_time

longest_trajtime_len, smallest_trajtime = find_trajtimelist()

def batch_timelist_ground_truth(valiortest = None):
    time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
    if valiortest == 'vali':
        time_list_int = time_list_int[10000:14000]   # based dataset and "validation or test" (train:validation:test = 1w:4k:1.6w)
    elif valiortest == 'test':
        time_list_int = time_list_int[14000:30000]

    sample_list = time_list_int[:5000]  # m*n matrix distance, m and n can be set by yourself

    pool = Pool(processes=19)
    for i in range(len(sample_list)+1):
        if i!=0 and i%50==0:
            pool.apply_async(timelist_distance, (i, sample_list[i-50:i], time_list_int, valiortest))
    pool.close()
    pool.join()

    return len(sample_list)

def merge_timelist_ground_truth(sample_len, valiortest):
    res = []
    for i in range(sample_len+1):
        if i!=0 and i%50==0:
            res.append(np.load('./ground_truth/{}/{}/{}_batch/{}_temporal_distance_{}.npy'.format(dataset, str(config["distance_type"]), valiortest, str(config["distance_type"]), str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/{}/{}_temporal_distance.npy'.format(dataset, str(config["distance_type"]), valiortest), res)

def timelist_distance(k, sample_list = [[]], test_list = [[]], valiortest=None):

    all_dis_list = []
    for sample in sample_list:
        one_dis_list = []
        for traj in test_list:
            if str(config["distance_type"]) == 'TP':
                one_dis_list.append(TP_dis(sample, traj))
            elif str(config["distance_type"]) == 'DITA':
                one_dis_list.append(DITA_dis(sample, traj))
            elif str(config["distance_type"]) == 'discret_frechet':
                one_dis_list.append(frechet_dis(sample, traj))
            elif str(config["distance_type"]) == 'LCRS':
                one_dis_list.append(LCRS_dis(sample, traj))
            elif str(config["distance_type"]) == 'NetERP':
                one_dis_list.append(NetERP_dis(sample, traj))
        all_dis_list.append(np.array(one_dis_list))

    all_dis_list = np.array(all_dis_list)
    p = './ground_truth/{}/{}/{}_batch/'.format(dataset, str(config["distance_type"]), valiortest)
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('./ground_truth/{}/{}/{}_batch/{}_temporal_distance_{}.npy'.format(dataset, str(config["distance_type"]), valiortest, str(config["distance_type"]), str(k)), all_dis_list)

    print('complete: ' + str(k))

@numba.jit(nopython=True, fastmath=True)
def TP_dis(list_a = [] , list_b = []):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            temp = abs(tr1[i]-tr2[j])
            if temp < mindis:
                mindis = temp
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            temp = abs(tr2[i]-tr1[j])
            if temp < mindis:
                mindis = temp
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1,max2))

@numba.jit(nopython=True, fastmath=True)
def DITA_dis(list_a = [], list_b = []):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M, N))
    cost[0, 0] = abs(tr1[0]-tr2[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + abs(tr1[i]-tr2[0])
    for i in range(1, N):
        cost[0, i] = cost[0, i - 1] + abs(tr1[0]-tr2[i])
    for i in range(1, M):
        for j in range(1, N):
            small = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(small) + abs(tr1[i]-tr2[j])
    return int(cost[M - 1, N - 1])

@numba.jit(nopython=True, fastmath=True)
def frechet_dis(list_a = [], list_b = []):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    c = np.zeros((M + 1, N + 1))
    c[0, 0] = abs(tr1[0]-tr2[0])
    for i in range(1, M):
        temp = abs(tr1[i]-tr2[0])
        if temp > c[i - 1][0]:
            c[i][0] = temp
        else:
            c[i][0] = c[i - 1][0]
    for i in range(1, N):
        temp = abs(tr2[i]-tr1[0])
        if temp > c[0][i - 1]:
            c[0][i] = temp
        else:
            c[0][i] = c[0][i - 1]
    for i in range(1, M):
        for j in range(1, N):
            c[i, j] = max(abs(tr1[i]-tr2[j]), min(c[i - 1][j - 1], c[i - 1][j], c[i][j - 1]))

    return int(c[M - 1, N - 1])

@numba.jit(nopython=True, fastmath=True)
def LCRS_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if abs(list_a[i] - list_b[j]) <= 3600:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1] == 0:
        return longest_trajtime_len*2
    else:
        return (lena + lenb - c[-1][-1])/float(c[-1][-1])

@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)

    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(lena + 1):
        edit[i][0] = i * smallest_trajtime
    for i in range(lenb + 1):
        edit[0][i] = i * smallest_trajtime

    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            edit[i][j] = min(edit[i - 1][j] + list_a[i-1] - smallest_trajtime, edit[i][j - 1] + list_b[j-1] - smallest_trajtime, edit[i - 1][j - 1] + abs(list_a[i-1] - list_b[j-1]))

    return edit[-1][-1]

if __name__ == '__main__':
    sample_len = batch_timelist_ground_truth(valiortest='vali')
    merge_timelist_ground_truth(sample_len=sample_len, valiortest='vali')
    sample_len = batch_timelist_ground_truth(valiortest='test')
    merge_timelist_ground_truth(sample_len=sample_len, valiortest='test')
