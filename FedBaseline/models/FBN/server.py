
import numpy as np
import copy
import torch


class Sever(object):
    w_glob = []
    loss_train = []  # 训练损失

    def __init__(self, args, w_glob):
        self.args = args
        self.w_glob = w_glob

    def aggregate(self, weight_locals, w_locals, avg_momentum=0.1):
        w_avg = copy.deepcopy(self.w_glob)
        weight_sum = sum(weight_locals)

        temp_running_mean = [0 for i in range(len(w_locals))]
        temp_running_var = [0 for i in range(len(w_locals))]
        temp_mean = 0
        temp_var = 0
        for k in w_avg.keys():
            w_avg[k] = 0
            if ('running_num' in k):
                w_avg[k] = torch.zeros(1).to(self.args.device)
            elif ('running_mean' in k):
                for i in range(len(w_locals)):
                    temp_running_mean[i] = w_locals[i][k]
                for i in range(len(w_locals)):
                    w_avg[k] += weight_locals[i] * temp_running_mean[i]
                w_avg[k] = w_avg[k] / weight_sum
                w_avg[k] = avg_momentum * w_avg[k] + (1 - avg_momentum) * self.w_glob[k]
                temp_mean = w_avg[k]
            elif ('running_var' in k):
                for i in range(len(w_locals)):
                    temp_running_var[i] = w_locals[i][k]
                for i in range(len(w_locals)):
                    w_avg[k] += weight_locals[i] * temp_running_var[i] + weight_locals[i] * ((
                            temp_running_mean[i] - temp_mean) ** 2)
                w_avg[k] = w_avg[k] / (weight_sum - 1)
                w_avg[k] = avg_momentum * w_avg[k] + (1 - avg_momentum) * self.w_glob[k]
                temp_var = w_avg[k]
            elif ('global_mean' in k):
                w_avg[k] = temp_mean
            elif ('global_var' in k):
                w_avg[k] = temp_var
            else:
                for i in range(len(w_locals)):
                    w_avg[k] += torch.mul(w_locals[i][k], weight_locals[i] / weight_sum)
        return w_avg
