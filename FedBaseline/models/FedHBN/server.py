import numpy as np
import copy
import torch
from torch import Tensor


class Sever(object):
    w_glob = []
    loss_train = []  # 训练损失

    def __init__(self, args, w_glob):
        self.args = args
        self.w_glob = w_glob

    def aggregate_without_buffer(self, weight_locals, w_locals):
        weight_sum = sum(weight_locals)
        w_avg = copy.deepcopy(self.w_glob)
        for k in w_avg.keys():
            w_avg[k] = 0
            if ('running_num' in k):
                w_avg[k] = self.w_glob[k]
            elif ('running_mean' in k):
                w_avg[k] = self.w_glob[k]
            elif ('running_var' in k):
                w_avg[k] = self.w_glob[k]
            elif ('global_mean' in k):
                w_avg[k] = self.w_glob[k]
            elif ('global_var' in k):
                w_avg[k] = self.w_glob[k]
            else:
                for i in range(len(w_locals)):
                    w_avg[k] += torch.mul(w_locals[i][k], weight_locals[i])
                w_avg[k] = w_avg[k] / weight_sum
        return w_avg

    def aggregate_only_buffer(self, weight_locals, w_locals, avg_momentum=1):
        w_avg = copy.deepcopy(self.w_glob)
        temp_running_num = [0 for i in range(len(w_locals))]
        temp_running_mean = [0 for i in range(len(w_locals))]
        temp_running_var = [0 for i in range(len(w_locals))]
        temp_mean = 0
        temp_num = 0

        for k in w_avg.keys():
            w_avg[k] = 0
            if ('running_num' in k):
                temp_num = 0
                for i in range(len(w_locals)):
                    temp_running_num[i] = w_locals[i][k]
                    temp_num += w_locals[i][k]
                w_avg[k] = torch.tensor([-1])
            elif ('running_mean' in k):
                for i in range(len(w_locals)):
                    temp_running_mean[i] = w_locals[i][k]
                w_avg[k] = torch.zeros(len(self.w_glob[k]))
            elif ('running_var' in k):
                for i in range(len(w_locals)):
                    temp_running_var[i] = w_locals[i][k]
                w_avg[k] = torch.ones(len(self.w_glob[k]))
            elif ('global_mean' in k):
                for i in range(len(w_locals)):
                    w_avg[k] += temp_running_num[i] * temp_running_mean[i]
                w_avg[k] = w_avg[k] / temp_num
                temp_mean = w_avg[k]
                w_avg[k] = avg_momentum * w_avg[k] + (1 - avg_momentum) * self.w_glob[k]
            elif ('global_var' in k):
                for i in range(len(w_locals)):
                    w_avg[k] += temp_running_num[i] * temp_running_var[i] \
                                + temp_running_num[i] * (temp_running_mean[i] - temp_mean) ** 2
                w_avg[k] = w_avg[k] / (temp_num - 1)
                w_avg[k] = avg_momentum * w_avg[k] + (1 - avg_momentum) * self.w_glob[k]
            else:
                w_avg[k] = self.w_glob[k]
        return w_avg

