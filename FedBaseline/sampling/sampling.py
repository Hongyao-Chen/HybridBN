from FedBaseline.models.FedAvg.clients import Client as AvgClient
from FedBaseline.models.FixBN.clients import Client as FixBNClient
from FedBaseline.models.FedHBN.clients import Client as HBNClient
from FedBaseline.models.FBN.clients import Client as FBNClient
from FedBaseline.models.FedFN.clients import Client as FNClient

import matplotlib.pyplot as plt


def DatasetDivide(args, dataset, client=None):
    if client is None:
        client = []

    num_users = args.num_users
    num_items = int(len(dataset) / num_users)
    if args.fed_name == 'fedhbn':
        client = [HBNClient(args) for i in range(num_users)]
    if args.fed_name == 'fedavg':
        client = [AvgClient(args) for i in range(num_users)]
    if args.fed_name == 'fixbn':
        client = [FixBNClient(args) for i in range(num_users)]
    if args.fed_name == 'fbn':
        client = [FBNClient(args) for i in range(num_users)]
    if args.fed_name == 'fedfn':
        client = [FNClient(args) for i in range(num_users)]
    all_idxs = [i for i in range(len(dataset))]

    if args.iid:
        for i in range(num_users):
            client[i].idxs = list(set(np.random.choice(all_idxs, num_items, replace=False)))
            client[i].datasize = len(client[i].idxs)
            all_idxs = list(set(all_idxs) - set(client[i].idxs))
    else:
        if args.dirichlet_alpha != 0.0:
            n_classes = args.num_classes
            label_distribution = np.random.dirichlet([args.dirichlet_alpha] * num_users, n_classes)
            labels = np.concatenate([np.array(dataset.targets)], axis=0)
            class_idcs = [np.argwhere(labels == y).flatten()
                          for y in range(n_classes)]
            client_idcs = [[] for _ in range(num_users)]
            for k_idcs, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(k_idcs,
                                                  (np.cumsum(fracs)[:-1] * len(k_idcs)).
                                                          astype(int))):
                    client_idcs[i] += [idcs]
            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            for i in range(args.num_users):
                client[i].idxs = client_idcs[i]
                client[i].datasize = len(client_idcs[i])

        else:
            num_shards = args.num_users * args.local_num_shards
            num_imgs = int(len(dataset) / num_shards)
            idx_shard = [i for i in range(num_shards)]
            idxs = np.arange(num_shards * num_imgs)
            labels = dataset.targets


            idxs_labels = np.vstack((idxs, labels))
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
            idxs = idxs_labels[0, :]

            for i in range(num_users):
                rand_set = set(np.random.choice(idx_shard, args.local_num_shards, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                client[i].idxs = np.array([], dtype='int64')
                for rand in rand_set:
                    client[i].idxs = np.concatenate((client[i].idxs, idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                                                    axis=0)
                    client[i].datasize = len(client[i].idxs)
    return client



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def ClientShow(args, client, train_data, test_data):
    n_classes = args.num_classes
    n_clients = args.num_users
    labels = np.concatenate(
        [[d[1] for d in train_data], [d[1] for d in test_data]], axis=0)
    client_idcs = []

    for c in client:
        client_idcs.append(c.idxs)

    label_distribution = np.zeros((n_classes, n_clients), dtype=int)
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx], c_id] += 1

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(label_distribution, annot=False, cmap='Blues',
                     xticklabels=[f'{i}' for i in range(n_clients)],
                     yticklabels=[f'{i}' for i in range(n_classes)],
                     cbar_kws={'label': 'Number of Samples'})
    cbar = ax.collections[0].colorbar

    cbar.ax.tick_params(labelsize=30)
    plt.xticks(range(0, n_clients, 10), [f'{i}' for i in range(0, n_clients, 10)])  # 设置y轴标签间隔
    plt.xlabel("Client ID", fontname='Times New Roman', fontsize=45)
    plt.ylabel("Class Label", fontname='Times New Roman', fontsize=45)
    plt.xticks(fontname='Times New Roman', fontsize=30)
    plt.yticks(fontname='Times New Roman', fontsize=30)

    plt.show()

    return torch.Tensor(label_distribution).T
