import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import random
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import spatial_similarity as spatial_com
import temporal_similarity as temporal_com
import pickle
import yaml
import os

random.seed(1933)
config = yaml.safe_load(open('config.yaml'))

def load_netowrk(dataset):
    """
    load road network from file with Pytorch geometric data object
    :param dataset: the city name of road network
    :return: Pytorch geometric data object of the graph
    """
    edge_path = "./data/" + dataset + "/road/edge_weight.csv"
    node_embedding_path = "./data/" + dataset + "/node_features.npy"

    node_embeddings = np.load(node_embedding_path)
    df_dege = pd.read_csv(edge_path, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    edge_attr = df_dege["length"].to_numpy()

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    print("node embeddings shape: ", node_embeddings.shape)
    print("edge_index shap: ", edge_index.shape)
    print("edge_attr shape: ", edge_attr.shape)

    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    return road_network

class DataLoader():
    def __init__(self):
        self.kseg = config["kseg"]
        self.train_set = 10000
        self.vali_set = 14000
        self.test_set = 30000

    def load(self, load_part):
        # split train, vali, test set
        node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
        time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
        d2vec_list_int = np.load(str(config["shuffle_d2vec_file"]), allow_pickle=True)

        train_set = self.train_set
        vali_set = self.vali_set
        test_set = self.test_set

        if load_part=='train':
            return node_list_int[:train_set], time_list_int[:train_set], d2vec_list_int[:train_set]
        if load_part=='vali':
            return node_list_int[train_set:vali_set], time_list_int[train_set:vali_set], d2vec_list_int[train_set:vali_set]
        if load_part=='test':
            return node_list_int[vali_set:test_set], time_list_int[vali_set:test_set], d2vec_list_int[vali_set:test_set]

    def ksegment_ST(self):
        # Simplify the trajectory
        kseg_coor_trajs = np.load(str(config["shuffle_kseg_file"]), allow_pickle=True)[:self.train_set]
        time_trajs = np.load(str(config["shuffle_time_file"]), allow_pickle=True)[:self.train_set]

        kseg_time_trajs = []
        for t in time_trajs:
            kseg_time = []
            seg = len(t) // self.kseg
            t = np.array(t)
            for i in range(self.kseg):
                if i == self.kseg - 1:
                    kseg_time.append(np.mean(t[i * seg:]))
                else:
                    kseg_time.append(np.mean(t[i * seg:i * seg + seg]))
            kseg_time_trajs.append(kseg_time)
        kseg_time_trajs = np.array(kseg_time_trajs)
        print(kseg_time_trajs.shape)
        print(kseg_time_trajs[0])

        max_lat = 0
        max_lon = 0
        for traj in kseg_coor_trajs:
            for t in traj:
                if max_lat<t[0]:
                    max_lat = t[0]
                if max_lon<t[1]:
                    max_lon = t[1]
        kseg_coor_trajs = kseg_coor_trajs/[max_lat,max_lon]
        kseg_coor_trajs = kseg_coor_trajs.reshape(-1,self.kseg*2)
        kseg_time_trajs = kseg_time_trajs/np.max(kseg_time_trajs)

        kseg_ST = np.concatenate((kseg_coor_trajs, kseg_time_trajs), axis=1)
        print("kseg_ST len: ", len(kseg_ST))
        print("kseg_ST shape: ", kseg_ST.shape)

        return kseg_ST

    def get_triplets(self):
        train_node_list, train_time_list, train_d2vec_list = self.load(load_part='train')

        sample_train2D = self.ksegment_ST()

        ball_tree = BallTree(sample_train2D)

        anchor_index = list(range(len(train_node_list)))
        random.shuffle(anchor_index)

        apn_node_triplets = []
        apn_time_triplets = []
        apn_d2vec_triplets = []
        for j in range(1,1001):
            for i in anchor_index:
                dist, index = ball_tree.query([sample_train2D[i]], j+1)  # k nearest neighbors
                p_index = list(index[0])
                p_index = p_index[-1]

                p_sample = train_node_list[p_index]  # positive sample
                n_index = random.randint(0,len(train_node_list)-1)
                n_sample = train_node_list[n_index]  # negative sample
                a_sample = train_node_list[i]  # anchor sample

                ok = True
                if str(config["distance_type"]) == "TP":
                    if spatial_com.TP_dis(a_sample,p_sample)==-1 or spatial_com.TP_dis(a_sample,n_sample)==-1:
                        ok = False
                elif str(config["distance_type"]) == "DITA":
                    if spatial_com.DITA_dis(a_sample,p_sample)==-1 or spatial_com.DITA_dis(a_sample,n_sample)==-1:
                        ok = False
                elif str(config["distance_type"]) == "discret_frechet":
                    if spatial_com.frechet_dis(a_sample,p_sample)==-1 or spatial_com.frechet_dis(a_sample,n_sample)==-1:
                        ok = False
                elif str(config["distance_type"]) == "LCRS":
                    if spatial_com.LCRS_dis(a_sample, p_sample) == spatial_com.longest_traj_len*2 or temporal_com.LCRS_dis(a_sample,p_sample) == temporal_com.longest_trajtime_len*2:
                        ok = False
                elif str(config["distance_type"]) == "NetERP":
                    if spatial_com.NetERP_dis(a_sample,p_sample)==-1 or spatial_com.NetERP_dis(a_sample,n_sample)==-1:
                        ok = False

                if ok:
                    apn_node_triplets.append([a_sample, p_sample, n_sample])   # nodelist

                    p_sample = train_time_list[p_index]
                    n_sample = train_time_list[n_index]
                    a_sample = train_time_list[i]

                    apn_time_triplets.append([a_sample, p_sample, n_sample])   # timelist

                    p_sample = train_d2vec_list[p_index]
                    n_sample = train_d2vec_list[n_index]
                    a_sample = train_d2vec_list[i]

                    apn_d2vec_triplets.append([a_sample, p_sample, n_sample])  # d2veclist
                if len(apn_node_triplets)==len(train_node_list)*2:    # based on the num of train triplets we need
                    break
            if len(apn_node_triplets) == len(train_node_list)*2:
                break
        print("complete: sample")
        print(len(apn_time_triplets))
        print(apn_node_triplets[0])
        p = './data/{}/triplet/{}/'.format(dataset, str(config["distance_type"]))
        if not os.path.exists(p):
            os.makedirs(p)
        pickle.dump(apn_node_triplets,open(str(config["path_node_triplets"]),'wb'))
        pickle.dump(apn_time_triplets, open(str(config["path_time_triplets"]), 'wb'))
        pickle.dump(apn_d2vec_triplets, open(str(config["path_d2vec_triplets"]), 'wb'))

    def return_triplets_num(self):
        apn_node_triplets = pickle.load(open(str(config["path_node_triplets"]), 'rb'))
        return len(apn_node_triplets)

def triplet_groud_truth():
    apn_node_triplets = pickle.load(open(str(config["path_node_triplets"]),'rb'))
    apn_time_triplets = pickle.load(open(str(config["path_time_triplets"]),'rb'))
    com_max_s = []
    com_max_t = []
    for i in range(len(apn_time_triplets)):
        if str(config["distance_type"]) == "TP":
            ap_s = spatial_com.TP_dis(apn_node_triplets[i][0],apn_node_triplets[i][1])
            an_s = spatial_com.TP_dis(apn_node_triplets[i][0],apn_node_triplets[i][2])
            com_max_s.append([ap_s,an_s])
            ap_t = temporal_com.TP_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
            an_t = temporal_com.TP_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
            com_max_t.append([ap_t,an_t])
        elif str(config["distance_type"]) == "DITA":
            ap_s = spatial_com.DITA_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.DITA_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
            ap_t = temporal_com.DITA_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
            an_t = temporal_com.DITA_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
            com_max_t.append([ap_t, an_t])
        elif str(config["distance_type"]) == "discret_frechet":
            ap_s = spatial_com.frechet_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.frechet_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
            ap_t = temporal_com.frechet_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
            an_t = temporal_com.frechet_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
            com_max_t.append([ap_t, an_t])
        elif str(config["distance_type"]) == "LCRS":
            ap_s = spatial_com.LCRS_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.LCRS_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
            ap_t = temporal_com.LCRS_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
            an_t = temporal_com.LCRS_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
            com_max_t.append([ap_t, an_t])
        elif str(config["distance_type"]) == "NetERP":
            ap_s = spatial_com.NetERP_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.NetERP_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
            ap_t = temporal_com.NetERP_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
            an_t = temporal_com.NetERP_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
            com_max_t.append([ap_t, an_t])

    com_max_s = np.array(com_max_s)
    com_max_t = np.array(com_max_t)

    if str(config["dataset"]) == "tdrive":
        if str(config["distance_type"]) == "TP": coe = 8
        elif str(config["distance_type"]) == "DITA": coe = 32
        elif str(config["distance_type"]) == "LCRS": coe = 4
        elif str(config["distance_type"]) == "NetERP": coe = 8
    if str(config["dataset"]) == "rome":
        if str(config["distance_type"]) == "TP": coe = 8
        elif str(config["distance_type"]) == "DITA": coe = 16
        elif str(config["distance_type"]) == "LCRS": coe = 2
        elif str(config["distance_type"]) == "NetERP": coe = 8

    # Fix effects of extreme values
    com_max_s = com_max_s/np.max(com_max_s)*coe
    com_max_t = com_max_t/np.max(com_max_t)*coe

    train_triplets_dis = (com_max_s+com_max_t)/2

    np.save(str(config["path_triplets_truth"]), train_triplets_dis)
    print("complete: triplet groud truth")
    print(train_triplets_dis[0])

def test_merge_st_dis(valiortest = None):
    s = np.load('./ground_truth/{}/{}/{}_spatial_distance.npy'.format(str(config["dataset"]), str(config["distance_type"]), valiortest))
    t = np.load('./ground_truth/{}/{}/{}_temporal_distance.npy'.format(str(config["dataset"]), str(config["distance_type"]), valiortest))
    print(s.shape)

    unreach = {}
    for i, dis in enumerate(s):
        tmp = []
        for j, und in enumerate(dis):
            if und == -1:
                tmp.append(j)
        if len(tmp)>0:
            unreach[i] = tmp

    s = s/np.max(s)
    t = t/np.max(t)

    st = (s+t)/2

    for i in unreach.keys():
        st[i][unreach[i]]=-1

    if valiortest == 'vali':
        np.save(str(config["path_vali_truth"]), st)
    else:
        np.save(str(config["path_test_truth"]), st)

    print("complete: merge_st_distance")

class batch_list():
    def __init__(self, batch_size):
        self.apn_node_triplets = np.array(pickle.load(open(str(config["path_node_triplets"]), 'rb')))
        self.apn_d2vec_triplets = np.array(pickle.load(open(str(config["path_d2vec_triplets"]), 'rb')))
        self.batch_size = batch_size
        self.start = len(self.apn_node_triplets)    # ordered is '0' ; reverse is 'maxsize'

    def getbatch_one(self):
        '''
        # batch random
        index = list(range(len(self.apn_node_triplets)))
        random.shuffle(index)
        batch_index = random.sample(index, self.batch_size)

        # batch ordered
        if self.start + self.batch_size > len(self.apn_node_triplets):
            self.start = 0
        batch_index = list(range(self.start, self.start + self.batch_size))
        self.start += self.batch_size
        '''

        # batch reverse
        if self.start - self.batch_size < 0:
            self.start = len(self.apn_node_triplets)
        batch_index = list(range(self.start - self.batch_size, self.start))
        self.start -= self.batch_size

        node_list = self.apn_node_triplets[batch_index]
        time_list = self.apn_d2vec_triplets[batch_index]

        a_node_batch = []
        a_time_batch = []
        p_node_batch = []
        p_time_batch = []
        n_node_batch = []
        n_time_batch = []
        for tri1 in node_list:
            a_node_batch.append(tri1[0])
            p_node_batch.append(tri1[1])
            n_node_batch.append(tri1[2])
        for tri2 in time_list:
            a_time_batch.append(tri2[0])
            p_time_batch.append(tri2[1])
            n_time_batch.append(tri2[2])

        return a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index

if __name__ == "__main__":
    # data = DataLoader()
    # data.get_triplets()
    # triplet_groud_truth()
    test_merge_st_dis(valiortest='vali')
    test_merge_st_dis(valiortest='test')


