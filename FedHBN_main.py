import pickle
import time

import torch
import matplotlib.pyplot as plt

from FedBaseline.models.FedHBN.server import Sever
from FedBaseline.models.Nets import set_sever_net
from FedBaseline.models.Test import test_img, SaveMyResult
from FedBaseline.options.options import args_parser
from FedBaseline.sampling.dataLoader import dataLoader
from FedBaseline.sampling.sampling import ClientShow, DatasetDivide
import copy
import numpy as np

if __name__ == '__main__':
    args = args_parser()
    args.fed_name = 'fedhbn'
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    save_model_path = 'Savedmodel/' + args.model + '_' + args.fed_name + "_" + str(
        args.dirichlet_alpha) + '_' + args.dataset

    dataset_train, dataset_test = dataLoader(args)
    clients = DatasetDivide(args, dataset_train)
    # ClientShow(args, clients, dataset_train, dataset_test)
    net_glob = set_sever_net(args)

    print("Model :", net_glob)
    ACC = []
    net_glob.train()
    best_acc = 0.0
    w0 = net_glob.state_dict()
    server = Sever(args=args, w_glob=w0)

    #################### Train ##########################################
    for iter in range(args.communication_round + 1):
        start_time = time.time()
        m = max(int(args.frac * args.num_users), 1)
        w_locals = []
        buffer_w_locals = []
        loss_locals = []
        weight_locals = []
        idxs_client = np.random.choice(range(args.num_users), m, replace=False)
        idxs_client = np.sort(idxs_client)

        ######## buffer update ##########
        for idx in idxs_client:
            if clients[idx].datasize >= args.local_batch:
                buffer_w = clients[idx].buffer_update(dataset=dataset_train,
                                                      net=copy.deepcopy(net_glob).to(args.device))
                buffer_w_locals.append(copy.deepcopy(buffer_w))

        ######## train clients ##########
        for idx in idxs_client:
            if clients[idx].datasize >= args.local_batch:
                w, loss = clients[idx].AVGupdate(dataset=dataset_train, net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                weight_locals.append(clients[idx].datasize)
        ######## aggregation ##########
        server.w_glob = server.aggregate_without_buffer(weight_locals, w_locals)
        server.w_glob = server.aggregate_only_buffer(weight_locals, buffer_w_locals, avg_momentum=0.002)
        net_glob.load_state_dict(server.w_glob)
        ###########################################################

        end_time = time.time()
        epoch_time = end_time - start_time
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, selected clients {}'.format(iter, idxs_client))
        print('clients loss: {}, Average loss {:.3f}, Time {:.3f}'.format([round(x, 3) for x in loss_locals], loss_avg,
                                                                          epoch_time))
        server.loss_train.append(loss_avg)
        args.lr = args.lr * args.gamma

        ############# testing #################
        if iter % args.verbose == 0:
            buffer_w_locals = []
            for idx in idxs_client:
                if clients[idx].datasize >= args.local_batch:
                    buffer_w = clients[idx].buffer_update(dataset=dataset_train,
                                                          net=copy.deepcopy(net_glob).to(args.device))
                    buffer_w_locals.append(copy.deepcopy(buffer_w))
            test_net = copy.deepcopy(net_glob)
            test_net.load_state_dict(server.aggregate_only_buffer(weight_locals, buffer_w_locals))
            acc_test, loss_test = test_img(test_net, dataset_test, args)
            print('test: Acc {:.3f}, Loss {:.3f}'.format(acc_test, loss_test))
            ACC.append(acc_test.item())
            if acc_test > best_acc:
                best_acc = acc_test
                best_loss = loss_test
                with open(save_model_path, 'wb') as f:
                    pickle.dump(net_glob, f)
        ########################################

    SaveMyResult(args, server.loss_train, ACC, best_acc, save_model_path)
