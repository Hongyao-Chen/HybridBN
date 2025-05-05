import copy

import torch
from torch import nn
from torch.nn import BatchNorm2d
from torch.optim import SGD
from torchvision.models import resnet18

class FederatedBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(FederatedBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.momentum = 0.1
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_num', torch.zeros(1))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('global_mean', torch.zeros(num_features))
        self.register_buffer('global_var', torch.ones(num_features))
    def forward(self, input):
        if self.training:
            mean = self.global_mean[None, :, None, None]
            var = self.global_var[None, :, None, None]
            curr_mean = input.mean(dim=[0, 2, 3])
            curr_var = input.var(dim=[0, 2, 3], unbiased=False)
            all_num = self.running_num
            all_mean = (self.running_num * self.running_mean + curr_mean.detach()) / (all_num+1)
            all_var = (self.running_num * self.running_var + curr_var.detach()) / (all_num+1)
            self.running_num = all_num + 1
            self.running_mean = all_mean
            self.running_var = all_var
        else:
            mean = self.global_mean[None, :, None, None]
            var = self.global_var[None, :, None, None]

        input = (input - mean.view(1, -1, 1, 1)) / (var.view(1, -1, 1, 1) + self.eps).sqrt()
        if self.affine:
            input = input * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return input