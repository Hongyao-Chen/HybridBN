
import numpy as np
import copy
import torch


class Sever(object):
    w_glob = []
    loss_train = []  # 训练损失

    def __init__(self, args, w_glob):
        self.args = args
        self.w_glob = w_glob

    def aggregate(self, weight_locals, w_locals):
        w_avg = copy.deepcopy(w_locals[0])
        weight_sum = sum(weight_locals)
        for k in w_avg.keys():
            w_avg[k] = 0
            for i in range(len(w_locals)):
                w_avg[k] += torch.mul(w_locals[i][k], weight_locals[i] / weight_sum)
        self.w_glob = w_avg
        return w_avg
