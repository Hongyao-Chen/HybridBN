import copy

import torch
from torch import nn
from torch.nn import BatchNorm2d, BatchNorm1d
from torch.optim import SGD
from torchvision.models import resnet18


class HybridBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(HybridBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.distribution_factor = nn.Parameter(torch.zeros(num_features))
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_num', torch.ones(1))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('global_mean', torch.zeros(num_features))
        self.register_buffer('global_var', torch.ones(num_features))

    def forward(self, input):
        if self.training:
            curr_num = len(input)
            curr_mean = input.mean(dim=[0, 2, 3])
            curr_var = input.var(dim=[0, 2, 3], unbiased=False)
            a = torch.sigmoid(self.distribution_factor)
            mean = (1 - a) * self.global_mean + a * curr_mean
            var = (1 - a) * self.global_var + a * curr_var
            if self.running_num != -1: # Start updating local statistical parameters
                all_num = self.running_num + curr_num
                all_mean = (self.running_num * self.running_mean + curr_num * mean.detach()) / all_num
                all_var = (self.running_var * self.running_num + var.detach() * curr_num + curr_num * (
                        mean.detach() - all_mean) ** 2) / all_num
                self.running_num = all_num
                self.running_mean = all_mean
                self.running_var = all_var
        else:
            mean = self.global_mean[None, :, None, None]
            var = self.global_var[None, :, None, None]
        input = (input - mean.view(1, -1, 1, 1)) / (var.view(1, -1, 1, 1) + self.eps).sqrt()

        if self.affine:
            input = input * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return input

def clear_bn_buffer(net):
    w = net.state_dict()
    w_clear = copy.deepcopy(w)
    for k in w_clear.keys():
        if (('running_num' in k) or ('running_mean' in k) or ('running_var' in k)):
            w_clear[k] = torch.zeros(len(w_clear[k]))
        else:
            w_clear[k] = w[k]
    net.load_state_dict(w_clear)
