import argparse
import numpy as np
import torch
import warnings

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def args_parser():
    parser = argparse.ArgumentParser()
    # federated train arguments
    parser.add_argument('--communication_round', type=int, default=500, help="rounds of communication T")
    parser.add_argument('--fed_name', type=str, help="name of federated learning")
    parser.add_argument('--epochs', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--gamma', type=float, default=0.998, help="learning decay rate gamma")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")

    # A suitable learning rate is crucial for different normalization methods with different local_batch
    # You can try several rounds first to explore the appropriate initial learning rate
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate n, a suitable learning rate is crucial for different normalization methods")
    parser.add_argument('--local_batch', type=int, default=4, help="local batch size: B")

    # dataset divide arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")    # cifar10 cifar100 tiny-imagenet
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    #cifar10=10 cifar100=100 tiny-imagenet=200
    parser.add_argument('--num_users', type=int, default=100, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--iid', default=False, action='store_true', help='whether i.i.d or not')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.6, help='Parameter of the distribution')
    parser.add_argument('--local_num_shards', type=int, default=1, help='number of client num_shards if dirichlet_alpha==0')

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', type=int, default=1, help='verbose print')
    parser.add_argument('--premodel', default=False, action='store_true', help='pre-trained model')# Only open to ResNet, VGG, googlenet and MobileNet
    # Please run the main function corresponding to the model. FedAvg_main can only run gn,bn and ln.
    parser.add_argument('--model', type=str, default='cnn-hbn', help='model name')
    # 'cnn-bn'
    # 'cnn-hbn'
    # 'cnn-gn'
    # 'cnn-ln'
    # 'cnn-fbn'
    # 'cnn-fn'
    # 'cnn-nn' Without any norm layer
    # 'resnet18-bn'
    # 'resnet18-gn'
    # 'resnet18-hbn'
    # 'vgg19-bn'
    # 'vgg19-hbn'
    # 'vgg19-gn'
    # 'googlenet-bn'
    # 'googlenet-hbn'
    # 'googlenet-gn'
    # 'vgg11-bn'
    # 'vgg11-hbn'
    # 'vgg11-gn'
    # 'mobilenet-bn'
    # 'mobilenet-hbn'
    # 'mobilenet-gn'
    # 'resnet50-bn'
    # 'resnet50-gn'
    # 'resnet50-hbn'

    args = parser.parse_args()
    return args
