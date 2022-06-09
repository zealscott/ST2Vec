import yaml
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import Node2Vec
from Model import Date2VecConvert
import time
import datetime
import random

random.seed(1953)
def prepare_dataset(trajfile, timefile, kseg = 5):
    """
    :param trajfile: map-matching result
    :param timefile: raw coor-timestamp file
    :param kseg: Simplify the trajectory to kseg
    """
    node_list = pd.read_csv(trajfile)
    node_list = node_list.Node_list
    time_list = pd.read_csv(timefile)
    time_list = time_list.Time_list

    node_list_int = []
    for nlist in node_list:
        tmp_list = []
        nlist = nlist[1:-1].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ',').split(',')
        for n in nlist:
            if n != '':
                tmp_list.append(int(n))
        node_list_int.append(tmp_list)
    node_list_int = np.array(node_list_int)

    time_list_int = []
    for tlist in time_list:
        tmp_list = []
        tlist = tlist[1:-1].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ',').split(',')
        for t in tlist:
            if t != '':
                tmp_list.append(int(t))
        time_list_int.append(tmp_list)
    time_list_int = np.array(time_list_int)

    df = pd.read_csv(trajfile)
    trajs = df.Coor_list

    coor_trajs = []
    for traj in trajs:
        traj = traj[1:-1].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ',').split(',')
        ts = []
        for s in traj:
            if s != '':
                ts.append(float(s))
        traj = np.reshape(ts, [-1, 2], order='C')
        coor_trajs.append(traj)

    kseg_coor_trajs = []
    for t in coor_trajs:
        kseg_coor = []
        seg = len(t) // kseg
        t = np.array(t)
        for i in range(kseg):
            if i == kseg - 1:
                kseg_coor.append(np.mean(t[i * seg:], axis=0))
            else:
                kseg_coor.append(np.mean(t[i * seg:i * seg + seg], axis=0))
        kseg_coor_trajs.append(kseg_coor)
    kseg_coor_trajs = np.array(kseg_coor_trajs)
    print("complete: ksegment")

    shuffle_index = list(range(len(node_list_int)))
    random.shuffle(shuffle_index)
    shuffle_index = shuffle_index[:50000]   # 5w size of dataset

    coor_trajs = np.array(coor_trajs)
    coor_trajs = coor_trajs[shuffle_index]

    kseg_coor_trajs = kseg_coor_trajs[shuffle_index]
    time_list_int = time_list_int[shuffle_index]
    node_list_int = node_list_int[shuffle_index]

    np.save(str(config["shuffle_coor_file"]), coor_trajs)
    np.save(str(config["shuffle_node_file"]), node_list_int)
    np.save(str(config["shuffle_time_file"]), time_list_int)
    np.save(str(config["shuffle_kseg_file"]), kseg_coor_trajs)

class Date2vec(nn.Module):
    def __init__(self):
        super(Date2vec, self).__init__()
        self.d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")
    def forward(self, time_seq):
        all_list = []
        for one_seq in time_seq:
            one_list = []
            for timestamp in one_seq:
                t = datetime.datetime.fromtimestamp(timestamp)
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
                x = torch.Tensor(t).float()
                embed = self.d2v(x)
                one_list.append(embed)

            one_list = torch.cat(one_list, dim=0)
            one_list = one_list.view(-1, 64)

            all_list.append(one_list.numpy().tolist())

        all_list = np.array(all_list)

        return all_list

def read_graph(dataset):
    """
    Read network edages from text file and return networks object
    :param file: input dataset name
    :return: edage index with shape (n,2)
    """
    dataPath = "./data/" + dataset
    edge = dataPath + "/road/edge_weight.csv"
    node = dataPath + "/road/node.csv"

    df_dege = pd.read_csv(edge, sep=',')
    df_node = pd.read_csv(node, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    num_node = df_node["node"].size

    print("{0} road netowrk has {1} edages.".format(config["dataset"], edge_index.shape[0]))
    print("{0} road netowrk has {1} nodes.".format(config["dataset"], num_node))

    return edge_index, num_node


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer):
    # Training with epoch iteration
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(100):
        loss = train(model, loader, optimizer)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            break
        else:
            last_loss = loss

@torch.no_grad()
def save_embeddings(model, num_nodes, dataset, device):
    model.eval()
    node_features = model(torch.arange(num_nodes, device=device)).cpu().numpy()
    np.save("./data/" + dataset + "/node_features.npy", node_features)
    print("Node embedding saved at: ./data/" + dataset + "/node_features.npy")
    return

if __name__ == "__main__":
    config = yaml.safe_load(open('config.yaml'))

    edge_index, num_node = read_graph(str(config["dataset"]))

    device = "cuda:" + str(config["cuda"])
    feature_size = config["feature_size"]
    walk_length = config["node2vec"]["walk_length"]
    context_size = config["node2vec"]["context_size"]
    walks_per_node = config["node2vec"]["walks_per_node"]
    p = config["node2vec"]["p"]
    q = config["node2vec"]["q"]

    edge_index = torch.LongTensor(edge_index).t().contiguous().to(device)

    model = Node2Vec(
        edge_index,
        embedding_dim=feature_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True,
        num_nodes=num_node
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    # Train until delta loss has been reached
    train_epoch(model, loader, optimizer)

    save_embeddings(model, num_node, str(config["dataset"]), device)

    prepare_dataset(trajfile=str(config["traj_file"]), timefile=str(config["time_file"]), kseg=config["kseg"])

    d2vec = Date2vec()
    timelist = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
    d2v = d2vec(timelist)
    print(len(d2v))
    np.save(str(config["shuffle_d2vec_file"]), d2v)
