from model_network import STTrajSimEncoder
import yaml
import torch
import data_utils
from lossfun import LossFun
import test_method
import time
import random


class STsim_Trainer(object):
    def __init__(self):
        config = yaml.safe_load(open('config.yaml'))

        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size = config["date2vec_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.concat = config["concat"]
        self.device = "cuda:" + str(config["cuda"])
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

        self.train_batch = config["train_batch"]
        self.test_batch = config["test_batch"]
        self.traj_file = str(config["traj_file"])
        self.time_file = str(config["time_file"])

        self.dataset = str(config["dataset"])
        self.distance_type = str(config["distance_type"])
        self.early_stop = config["early_stop"]

    def ST_eval(self, load_model=None):
        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)

        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)

            dataload = data_utils.DataLoader()
            road_network = data_utils.load_netowrk(self.dataset).to(self.device)

            with torch.no_grad():
                vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
                embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_batch=self.test_batch)
                acc = test_method.test_model(embedding_vali, isvali=False)
                print(acc)

    def ST_train(self, load_model=None, load_optimizer=None):

        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)

        dataload = data_utils.DataLoader()
        dataload.get_triplets()
        data_utils.triplet_groud_truth()

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = LossFun(self.train_batch, self.distance_type)

        net.to(self.device)
        lossfunction.to(self.device)

        road_network = data_utils.load_netowrk(self.dataset).to(self.device)

        bt_num = int(dataload.return_triplets_num() / self.train_batch)

        batch_l = data_utils.batch_list(batch_size=self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        lastepoch = '0'
        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch = load_model.split('/')[-1].split('_')[3]
            best_epoch = int(lastepoch)

        for epoch in range(int(lastepoch), self.epochs):
            net.train()
            s1 = time.time()
            for bt in range(bt_num):
                a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index = batch_l.getbatch_one()

                a_embedding = net(road_network, a_node_batch, a_time_batch)
                p_embedding = net(road_network, p_node_batch, p_time_batch)
                n_embedding = net(road_network, n_node_batch, n_time_batch)

                loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            s5 = time.time()
            print("train time: ", s5-s1)
            if epoch%2 == 0:
                net.eval()
                with torch.no_grad():
                    s6 = time.time()
                    vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali')
                    embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                                   test_traj=list(vali_node_list),
                                                                   test_time=list(vali_d2vec_list),
                                                                   test_batch=self.test_batch)
                    acc = test_method.test_model(embedding_vali, isvali=True)
                    s7 = time.time()
                    print("test time: ", s7-s6)
                    print('epoch:', epoch, acc[0], acc[1], acc[2], loss.item())

                    # save model
                    save_modelname = './model/{}_{}_2w_ST/{}_{}_epoch_{}_HR10_{}_HR50_{}_HR1050_{}_Loss_{}.pkl'.format(self.dataset, self.distance_type,
                        self.dataset, self.distance_type, str(epoch), acc[0], acc[1], acc[2], loss.item())
                    torch.save(net.state_dict(), save_modelname)

                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                    if epoch - best_epoch >= self.early_stop:
                        break
                    '''
                    save_optname = './optimizer/{}/tdrive_TP_2w_ST/{}_{}_epoch_{}.pkl'.format(self.dataset, self.dataset,
                                                                                              self.distance_type,
                                                                                              str(epoch))
                    torch.save(optimizer.state_dict(), save_optname)
                    '''


