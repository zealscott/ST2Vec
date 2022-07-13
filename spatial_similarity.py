from STmatching_distribution_ver import network_data
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
import collections
import os

random.seed(1998)
config = yaml.safe_load(open('config.yaml'))
dataset = str(config["dataset"])
dataset_point = config["pointnum"][str(config["dataset"])]

def find_longest_trajectory():
    longest_traj = 0
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
    for node_list in node_list_int:
        if len(node_list)>longest_traj:
            longest_traj = len(node_list)
    return longest_traj

longest_traj_len = find_longest_trajectory()
def batch_Point_distance():

    pool = Pool(processes=20)
    for i in range(dataset_point + 1):
        if i != 0 and i % 1000 == 0:
            pool.apply_async(parallel_point_com, (i, list(range(i - 1000, i))))
    pool.close()
    pool.join()

def merge_Point_distance():
    res = []
    for i in range(dataset_point + 1):
        if i != 0 and i % 1000 == 0:
            res.append(np.load('./ground_truth/{}/Point_dis_matrix_{}.npy'.format(dataset, str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset), res)

def parallel_point_com(i, id_list = []):
    batch_list = []

    for k in id_list:
        one_list = []
        if k in roadnetwork.nodes():
            length_list = nx.shortest_path_length(roadnetwork, source=k, weight='distance')
            for j in range(dataset_point):
                if (j in length_list.keys()) == True:
                    one_list.append(length_list[j])
                else:
                    one_list.append(-1)
            batch_list.append(np.array(one_list,dtype=np.float32))
        else:
            one_list = [-1 for j in range(dataset_point)]
            batch_list.append(np.array(one_list,dtype=np.float32))

    batch_list = np.array(batch_list,dtype=np.float32)
    p = './ground_truth/{}/'.format(dataset)
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('./ground_truth/{}/Point_dis_matrix_{}.npy'.format(dataset, str(i)), batch_list)

def generate_point_matrix():
    res = np.load('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset))
    return res

def generate_node_edge_interation():
    node_edge_dict = collections.defaultdict(set)
    edge = pd.read_csv('./data/{}/road/edge_weight.csv'.format(dataset))
    node_s, node_e = edge.s_node, edge.e_node

    for idx, (n_s, n_e) in enumerate(zip(node_s, node_e)):
        node_edge_dict[int(n_s)].add(idx)
        node_edge_dict[int(n_e)].add(idx)

    return node_edge_dict

def batch_similarity_ground_truth(valiortest = None):
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
    if valiortest == 'vali':
        node_list_int = node_list_int[10000:14000]  # based dataset and "validation or test"  (train:validation:test = 1w:4k:1.6w)
    elif valiortest == 'test':
        node_list_int = node_list_int[14000:30000]

    sample_list = node_list_int[:5000]  # m*n matrix distance, m and n can be set by yourself

    pool = Pool(processes=19)
    for i in range(len(sample_list)+1):
        if i!=0 and i%50==0:
            pool.apply_async(Traj_distance, (i,sample_list[i-50:i],node_list_int, valiortest))
    pool.close()
    pool.join()

    return len(sample_list)

def merge_similarity_ground_truth(sample_len, valiortest):
    res = []
    for i in range(sample_len+1):
        if i!=0 and i%50==0:
            res.append(np.load('./ground_truth/{}/{}/{}_batch/{}_spatial_distance_{}.npy'.format(dataset, str(config["distance_type"]),valiortest, str(config["distance_type"]), str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/{}/{}_spatial_distance.npy'.format(dataset, str(config["distance_type"]), valiortest), res)

def Traj_distance(k, sample_list = [[]], test_list = [[]], valiortest = None):
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
    np.save('./ground_truth/{}/{}/{}_batch/{}_spatial_distance_{}.npy'.format(dataset, str(config["distance_type"]), valiortest, str(config["distance_type"]), str(k)), all_dis_list)

    print('complete: ' + str(k))

distance_matrix = generate_point_matrix()  # This line of code should be commented out when executing the current "spatial_similarity.py" file, but needed at any other time.

@numba.jit(nopython=True, fastmath=True)
def TP_dis(list_a = [] , list_b = []):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            if distance_matrix[tr1[i]][tr2[j]] != -1:
                temp = distance_matrix[tr1[i]][tr2[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            if distance_matrix[tr2[i]][tr1[j]] != -1:
                temp = distance_matrix[tr2[i]][tr1[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1,max2))

@numba.jit(nopython=True, fastmath=True)
def DITA_dis(list_a = [], list_b = []):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M, N))
    tp = distance_matrix[tr1[0]][tr2[0]]
    if tp==-1:
        return -1
    cost[0, 0] = tp
    for i in range(1, M):
        tp = distance_matrix[tr1[i]][tr2[0]]
        if tp == -1:
            return -1
        cost[i, 0] = cost[i - 1, 0] + tp
    for i in range(1, N):
        tp = distance_matrix[tr1[0]][tr2[i]]
        if tp == -1:
            return -1
        cost[0, i] = cost[0, i - 1] + tp
    for i in range(1, M):
        for j in range(1, N):
            small = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            tp = distance_matrix[tr1[i]][tr2[j]]
            if tp == -1:
                return -1
            cost[i, j] = min(small) + tp
    return int(cost[M - 1, N - 1])

@numba.jit(nopython=True, fastmath=True)
def frechet_dis(list_a = [], list_b = []):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    c = np.zeros((M + 1, N + 1))
    tp = distance_matrix[tr1[0]][tr2[0]]
    if tp == -1:
        return -1
    c[0, 0] = tp
    for i in range(1, M):
        tp = distance_matrix[tr1[i]][tr2[0]]
        if tp == -1:
            return -1
        temp = tp
        if temp > c[i - 1][0]:
            c[i][0] = temp
        else:
            c[i][0] = c[i - 1][0]
    for i in range(1, N):
        tp = distance_matrix[tr1[0]][tr2[i]]
        if tp == -1:
            return -1
        temp = tp
        if temp > c[0][i - 1]:
            c[0][i] = temp
        else:
            c[0][i] = c[0][i - 1]
    for i in range(1, M):
        for j in range(1, N):
            tp = distance_matrix[tr1[i]][tr2[j]]
            if tp == -1:
                return -1
            c[i, j] = max(tp, min(c[i - 1][j - 1], c[i - 1][j], c[i][j - 1]))

    return int(c[M - 1, N - 1])

node_edge_dict = generate_node_edge_interation()
def LCRS_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if len(node_edge_dict[list_a[i]] & node_edge_dict[list_b[j]]) >= 1:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1]==0:
        return longest_traj_len*2
    else:
        return (lena + lenb - c[-1][-1])/float(c[-1][-1])

def hot_node():
    max_num = 0
    max_idx = 0
    for idx, nodes_interaction in enumerate(distance_matrix):
        nodes_interaction = np.array(nodes_interaction)
        x = len(nodes_interaction[nodes_interaction != -1])
        if x > max_num:
            max_num = x
            max_idx = idx
    print(max_num, max_idx)
    return max_idx

hot_node_id = hot_node()

@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)

    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(1, lena + 1):
        tp = distance_matrix[hot_node_id][list_a[i-1]]
        if tp == -1:
            return -1
        edit[i][0] = edit[i-1][0] + tp
    for i in range(1, lenb + 1):
        tp = distance_matrix[hot_node_id][list_b[i-1]]
        if tp == -1:
            return -1
        edit[0][i] = edit[0][i-1] + tp

    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            tp1 = distance_matrix[hot_node_id][list_a[i-1]]
            tp2 = distance_matrix[hot_node_id][list_b[j-1]]
            tp3 = distance_matrix[list_a[i-1]][list_b[j-1]]
            if tp1 == -1 or tp2 == -1 or tp3 == -1:
                return -1
            edit[i][j] = min(edit[i - 1][j] + tp1, edit[i][j - 1] + tp2, edit[i - 1][j - 1] + tp3)

    return edit[-1][-1]

if __name__ == '__main__':
    nx_vertice, nx_edge, vertice_dict, edge_dict, edge_dist, edge_dist_dict, roadnetwork = network_data()
    batch_Point_distance()
    merge_Point_distance()
    distance_matrix = generate_point_matrix()
    node_edge_dict = generate_node_edge_interation()
    sample_len = batch_similarity_ground_truth(valiortest='vali')
    merge_similarity_ground_truth(sample_len=sample_len, valiortest='vali')
    sample_len = batch_similarity_ground_truth(valiortest='test')
    merge_similarity_ground_truth(sample_len=sample_len, valiortest='test')





