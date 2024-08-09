import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, to_dense_adj
import torch.nn.functional as F
import torch_geometric.transforms as T


def load_fixed_splits(path):
    splits_lst = np.load(path, allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


def build_mask(labels, train_ratio, val_ratio, mode='balance'):
    num_nodes = labels.shape[0]
    num_classes = len(np.unique(labels))
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)
    num_test = num_nodes - num_train - num_val

    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    if mode == 'random':

        num_nodes = labels.shape[0]

        idx = np.random.permutation(num_nodes)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train + num_val]
        test_idx = idx[num_train + num_val:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        return torch.BoolTensor(train_mask), torch.BoolTensor(
            val_mask), torch.BoolTensor(test_mask)
    elif mode == 'balance':

        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            np.random.shuffle(idx)
            class_num_nodes = len(idx)
            train_idx = idx[:int(class_num_nodes * train_ratio)]
            val_idx = idx[int(class_num_nodes *
                              train_ratio):int(class_num_nodes *
                                               (train_ratio + val_ratio))]
            test_idx = idx[int(class_num_nodes * (train_ratio + val_ratio)):]

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
        return torch.BoolTensor(train_mask), torch.BoolTensor(
            val_mask), torch.BoolTensor(test_mask)
    elif mode == 'balance_48':

        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            np.random.shuffle(idx)
            class_num_nodes = len(idx)
            train_idx = idx[:int(class_num_nodes * 0.48)]
            val_idx = idx[int(class_num_nodes * 0.48):int(class_num_nodes *
                                                          0.8)]
            test_idx = idx[int(class_num_nodes * 0.8):]
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
        return torch.BoolTensor(train_mask), torch.BoolTensor(
            val_mask), torch.BoolTensor(test_mask)
    else:
        raise NotImplementedError


def normalize_adj(edge_index, num_nodes, norm='sys'):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

    if norm == 'sys':
        deg = scatter(edge_weight, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum") + \
              scatter(edge_weight, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_edge_weight = deg_inv_sqrt[
            edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]

    elif norm == 'row':
        deg_out = scatter(edge_weight,
                          edge_index[0],
                          dim=0,
                          dim_size=num_nodes,
                          reduce="sum")

        deg_out_inv = deg_out.pow(-1)
        deg_out_inv[deg_out_inv == float('inf')] = 0

        norm_edge_weight = deg_out_inv[edge_index[0]] * edge_weight

    elif norm == 'in_out_sys':
        deg_in = scatter(edge_weight,
                         edge_index[1],
                         dim=0,
                         dim_size=num_nodes,
                         reduce="sum")
        deg_out = scatter(edge_weight,
                          edge_index[0],
                          dim=0,
                          dim_size=num_nodes,
                          reduce="sum")

        deg_in_inv_sqrt = deg_in.pow(-0.5)
        deg_out_inv_sqrt = deg_out.pow(-0.5)
        deg_in_inv_sqrt[deg_in_inv_sqrt == float('inf')] = 0
        deg_out_inv_sqrt[deg_out_inv_sqrt == float('inf')] = 0

        norm_edge_weight = deg_in_inv_sqrt[
            edge_index[1]] * edge_weight * deg_out_inv_sqrt[edge_index[0]]
    elif norm == 'none':
        norm_edge_weight = edge_weight
    else:
        raise NotImplementedError
    adj = to_dense_adj(
        edge_index,
        edge_attr=norm_edge_weight,
    )[0]
    assert adj.shape[0] == adj.shape[1] == num_nodes

    return adj


def load_data(dataset: str,
              norm='row',
              mode='default',
              return_adj=True,
              run=0):
    assert norm in ['row', 'sys', 'in_out_sys', 'none']
    if dataset in ['chameleon', 'squirrel']:
        from torch_geometric.datasets import WikipediaNetwork

        data = WikipediaNetwork(root='./data',
                                name=dataset,
                                geom_gcn_preprocess=True)

    elif dataset in ['computers', 'photo']:
        from torch_geometric.datasets import Amazon
        data = Amazon(root='./data', name=dataset)
        masks = np.load('self_splits/{}.npz'.format(dataset))
        data.train_mask = torch.BoolTensor(masks['train'])
        data.val_mask = torch.BoolTensor(masks['val'])
        data.test_mask = torch.BoolTensor(masks['test'])
    elif dataset in ['film']:
        from torch_geometric.datasets import Actor
        data = Actor(root='./data')
    elif dataset in ['tolokers']:
        from torch_geometric.datasets import HeterophilousGraphDataset
        data = HeterophilousGraphDataset(root='./data',
                                         name=dataset,
                                         pre_transform=T.ToUndirected())
    elif dataset in ["deezer-europe"]:
        filename = './self_splits/deezer-europe.mat'
        from scipy.io import loadmat
        deezer = loadmat(filename)
        A, label, features = deezer['A'], deezer['label'], deezer['features']
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        node_feat = torch.tensor(features.todense(), dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long).squeeze()

        num_labels = len(np.unique(label))
        labels = label

        features = node_feat
        num_features = features.shape[1]

        split_idx_ls = load_fixed_splits(
            './self_splits/deezer-europe-splits.npy')
        split_idx = split_idx_ls[run]
        train_mask, val_mask, test_mask = split_idx['train'], split_idx[
            'valid'], split_idx['test']
        if return_adj:
            adj = normalize_adj(edge_index, features.shape[0], norm=norm)
            return adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
        else:
            return edge_index, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
    else:
        raise 'dataset not found'

    if mode == 'default':
        data.train_mask = data.train_mask[:, run]
        data.val_mask = data.val_mask[:, run]
        data.test_mask = data.test_mask[:, run]

    if dataset.startswith('synth'):
        pass
    else:
        labels = torch.LongTensor(data.y)
        edge_index = data.edge_index
        features = data.x
    features = torch.FloatTensor(features)
    features = F.normalize(features, p=1, dim=1)

    labels = labels - labels.min()

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    if mode == 'default':
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    elif mode == 'balance':
        train_mask, val_mask, test_mask = build_mask(labels, 0.6, 0.2, mode)
    elif mode == 'balance_48':
        train_mask, val_mask, test_mask = build_mask(labels, 0.48, 0.32, mode)

    if return_adj:
        adj = normalize_adj(edge_index, features.shape[0], norm=norm)
        return adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
    else:
        return edge_index, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
