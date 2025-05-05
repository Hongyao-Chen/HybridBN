import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tqdm import trange

from FedBaseline.models.Nets import set_sever_net
from FedBaseline.models.Test import test_img, SaveMyResult
from FedBaseline.options.options import args_parser

import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from FedBaseline.sampling.dataLoader import dataLoader

if __name__ == '__main__':
    args = args_parser()
    args.fed_name = 'centered'
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    client = []
    dataset_train, dataset_test = dataLoader(args)

    best_acc = 0.0
    save_model_path = 'Savedmodel/' + args.model + '_' + args.fed_name + '_' + args.dataset
    net_glob = set_sever_net(args)
    print("Model :", net_glob)
    net_glob.train()
    # init
    w0 = net_glob.s0tate_dict()

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    list_loss = []
    tbar = trange(args.communication_round)
    for epoch in tbar:
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            if batch_idx % args.verbose == 0:
                mesg = "\tEpoch: {}\t[{}/{}]\tloss:{:.3f}\t".format(epoch, batch_idx, len(train_loader) - 1, loss)
                tbar.set_description(mesg)
        loss_avg = sum(batch_loss) / len(batch_loss)
        scheduler.step()
        print('\nTrain loss:', loss_avg)
        if epoch % args.verbose == 0:
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('test: Acc {:.3f}, Loss {:.3f}'.format(acc_test, loss_test))
            if acc_test > best_acc:
                best_acc = acc_test
                with open(save_model_path, 'wb') as f:
                    pickle.dump(net_glob, f)
        list_loss.append(loss_avg)


    print('best test: Acc {:.3f}'.format(best_acc))
