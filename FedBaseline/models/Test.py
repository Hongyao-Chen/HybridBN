#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset=datatest, batch_size=args.bs)
    l = len(datatest)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= l
    accuracy = 100.00 * correct / l
    return accuracy, test_loss


def SaveMyResult(args,loss_train,ACC, best_acc, path, num=10):
    ACC10 = ACC[-1 * num:]
    tensor_data = torch.tensor(ACC10)
    mean = torch.mean(tensor_data)
    std = torch.std(tensor_data, unbiased=True)
    z_score = 1.96
    std_error = std / torch.sqrt(torch.tensor(num))
    confidence_interval = (torch.tensor([-z_score, z_score]) * std_error).tolist()
    with open('log.txt', 'a') as file:
        file.write(str(path) + '\n' +
                   str(loss_train) + '\n' +
                   str(args) + '\n' +
                   str(ACC) + '\n' +
                   str(ACC10) + '\n' +
                   'best:' + str(best_acc.item()) + '\n' +
                   'mean:' + str(mean.item()) + '\n' +
                   'std:' + str(std.item()) + '\n' +
                   '95% confidence_interval:' + str(confidence_interval) + '\n\n'
                   )
