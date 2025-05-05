import concurrent
import pickle
import threading
from threading import Thread

import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib
import matplotlib.pyplot as plt
from FedBaseline.models.FedAvg.server import Sever
from FedBaseline.models.Nets import set_sever_net
from FedBaseline.models.Test import test_img, SaveMyResult
from FedBaseline.options.options import args_parser
from FedBaseline.sampling.dataLoader import dataLoader
from FedBaseline.sampling.sampling import ClientShow, DatasetDivide
import copy
import numpy as np

if __name__ == '__main__':
    args = args_parser()
    args.fed_name = 'fedfn'
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    save_model_path = 'Savedmodel/' + args.model + '_' + args.fed_name + "_" + str(
        args.dirichlet_alpha) + '_' + args.dataset

    dataset_train, dataset_test = dataLoader(args)
    client = DatasetDivide(args, dataset_train)
    ClientShow(args, client, dataset_train, dataset_test)
    net_glob = set_sever_net(args)
    print("Model :", net_glob)
    ACC = []
    net_glob.train()
    best_acc = 0.0
    w0 = net_glob.state_dict()
    server = Sever(args=args, w_glob=w0)
    for iter in range(args.communication_round + 1):
        m = max(int(args.frac * args.num_users), 1)
        w_locals = []
        loss_locals = []
        weight_locals = []

        idxs_client = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_client:
            if client[idx].datasize >= args.local_batch:
                w, loss = client[idx].update(dataset=dataset_train, net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                weight_locals.append(client[idx].datasize)


        server.aggregate(weight_locals, w_locals)
        net_glob.load_state_dict(server.w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, selected client {}'.format(iter, idxs_client))
        print('client loss: {}, Average loss {:.3f}'.format([round(x, 3) for x in loss_locals], loss_avg))
        if iter % args.verbose == 0:
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('test: Acc {:.3f}, Loss {:.3f}'.format(acc_test, loss_test))
            ACC.append(acc_test.item())
            if acc_test > best_acc:
                best_acc = acc_test
                best_loss = loss_test
                with open(save_model_path, 'wb') as f:
                    pickle.dump(net_glob, f)
        server.loss_train.append(loss_avg)
        args.lr = args.lr * args.gamma

    print('best test: Acc {:.3f}, Loss {:.3f}'.format(best_acc, best_loss))


    SaveMyResult(args,server.loss_train, ACC, best_acc, save_model_path)
