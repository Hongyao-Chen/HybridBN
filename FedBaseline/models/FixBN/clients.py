from torch import nn

import copy
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random


from FedBaseline.models.Test import test_img


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client(object):
    idxs = None
    datasize = 0.0

    def __init__(self, args):
        self.args = args

    def update(self, dataset=None, net=None, T2=False):
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs), batch_size=self.args.local_batch, shuffle=True,
                                    drop_last=True)
        self.loss_func = nn.CrossEntropyLoss()
        net = net.to(self.args.device)
        if T2:
            net.eval()
        else:
            net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
