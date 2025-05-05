import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from FedBaseline.models.FBN.FBN import FederatedBatchNorm
from FedBaseline.models.FedHBN.MyBNtool import HybridBatchNorm


def set_sever_net(args):
    net_glob = None

    if 'cnn-bn' == args.model:
        net_glob = SimpleCNN(args=args)
        replace_bn_with_bn(net_glob)
    elif 'cnn-hbn' == args.model:
        net_glob = SimpleCNN(args=args)
        replace_bn_with_hbn(net_glob)
    elif 'cnn-gn' == args.model:
        net_glob = SimpleCNN(args=args)
        replace_bn_with_gn(net_glob)
    elif 'cnn-ln' == args.model:
        net_glob = SimpleCNN(args=args)
        replace_bn_with_ln(net_glob)
    elif 'cnn-fbn' == args.model:
        net_glob = SimpleCNN(args=args)
        replace_bn_with_fbn(net_glob)
    elif 'cnn-fn' == args.model:
        net_glob = SimpleCNN(args=args, fn=True)
        replace_bn_with_nn(net_glob)
    elif 'cnn-nn' == args.model:
        net_glob = SimpleCNN(args=args)
        replace_bn_with_nn(net_glob)
    elif 'resnet18-bn' == args.model:
        net_glob = models.resnet18(pretrained=args.premodel, num_classes=args.num_classes)
        replace_bn_with_bn(net_glob)
    elif 'resnet18-gn' == args.model:
        net_glob = models.resnet18(pretrained=args.premodel, num_classes=args.num_classes)
        replace_bn_with_gn(net_glob)
    elif 'resnet18-hbn' == args.model:
        net_glob = models.resnet18(pretrained=args.premodel, num_classes=args.num_classes)
        replace_bn_with_hbn(net_glob)
    elif 'vgg19-bn' == args.model:
        net_glob = models.vgg19_bn(pretrained=args.premodel)
        replace_bn_with_bn(net_glob)
    elif 'vgg19-hbn' == args.model:
        net_glob = models.vgg19_bn(pretrained=args.premodel)
        replace_bn_with_hbn(net_glob)
    elif 'vgg19-gn' == args.model:
        net_glob = models.vgg19_bn(pretrained=args.premodel)
        replace_bn_with_gn(net_glob)
    elif 'googlenet-bn' == args.model:
        net_glob = models.googlenet(pretrained=args.premodel)
        replace_bn_with_bn(net_glob)
    elif 'googlenet-hbn' == args.model:
        net_glob = models.googlenet(pretrained=args.premodel)
        replace_bn_with_hbn(net_glob)
    elif 'googlenet-gn' == args.model:
        net_glob = models.googlenet(pretrained=args.premodel)
        replace_bn_with_gn(net_glob)
    elif 'vgg11-bn' == args.model:
        net_glob = models.vgg11_bn(pretrained=args.premodel)
        replace_bn_with_bn(net_glob)
    elif 'vgg11-hbn' == args.model:
        net_glob = models.vgg11_bn(pretrained=args.premodel)
        replace_bn_with_hbn(net_glob)
    elif 'vgg11-gn' == args.model:
        net_glob = models.vgg11_bn(pretrained=args.premodel)
        replace_bn_with_gn(net_glob)
    elif 'mobilenet-bn' == args.model:
        net_glob = models.mobilenet_v2(pretrained=args.premodel)
        replace_bn_with_bn(net_glob)
    elif 'mobilenet-hbn' == args.model:
        net_glob = models.mobilenet_v2(pretrained=args.premodel)
        replace_bn_with_hbn(net_glob)
    elif 'mobilenet-gn' == args.model:
        net_glob = models.mobilenet_v2(pretrained=args.premodel)
        replace_bn_with_gn(net_glob)
    elif 'resnet50-bn' == args.model:
        net_glob = models.resnet50(pretrained=args.premodel)
        replace_bn_with_bn(net_glob)
    elif 'resnet50-gn' == args.model:
        net_glob = models.resnet50(pretrained=args.premodel)
        replace_bn_with_gn(net_glob)
    elif 'resnet50-hbn' == args.model:
        net_glob = models.resnet50(pretrained=args.premodel)
        replace_bn_with_hbn(net_glob)
    replace_Dropout_with_nn(net_glob)
    return net_glob.to(args.device)


def replace_Dropout_with_nn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            id = nn.Identity()
            setattr(model, name, id)
        else:
            # Recursively replace in child modules
            replace_Dropout_with_nn(module)
def replace_bn_with_fbn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            # Replace BN with FBN
            fbn = FederatedBatchNorm(num_features=num_features)
            setattr(model, name, fbn)
        else:
            # Recursively replace in child modules
            replace_bn_with_fbn(module)

def replace_bn_with_bn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            # Replace BN with FBN
            bn = nn.BatchNorm2d(num_features=num_features)
            setattr(model, name, bn)
        else:
            # Recursively replace in child modules
            replace_bn_with_bn(module)
def replace_bn_with_gn(model, group=2):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            # Replace BN with GN
            gn = nn.GroupNorm(num_groups=group, num_channels=num_features)
            setattr(model, name, gn)
        else:
            # Recursively replace in child modules
            replace_bn_with_gn(module)


def replace_bn_with_ln(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            # Replace BN with LN
            ln = nn.GroupNorm(num_groups=1, num_channels=num_features)
            setattr(model, name, ln)
        else:
            # Recursively replace in child modules
            replace_bn_with_ln(module)


def replace_bn_with_hbn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            cbn = HybridBatchNorm(num_features=num_features)
            setattr(model, name, cbn)
        else:
            # Recursively replace in child modules
            replace_bn_with_hbn(module)


def replace_bn_with_nn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            nonnn = nn.Identity()
            setattr(model, name, nonnn)
        else:
            # Recursively replace in child modules
            replace_bn_with_nn(module)

class SimpleCNN(nn.Module):
    def __init__(self, args, fn=False, momentum=0.1):
        self.fn = fn
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            nn.BatchNorm2d(16, momentum=momentum),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 16*32*32 -> 16*16*16
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1),  # 16*16*16 -> 32*16*16
            nn.BatchNorm2d(32, momentum=momentum),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),  # 32*8*8 -> 64*8*8
            nn.BatchNorm2d(64, momentum=momentum),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 64*8*8 -> 64*4*4
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        features = x.view(-1, 64 * 4 * 4)
        if self.fn:
            features = F.normalize(features, p=2, dim=1) * torch.sqrt(torch.tensor(features.shape[1]))
        x = self.fc1(features)
        x = self.fc2(x)

        return x






