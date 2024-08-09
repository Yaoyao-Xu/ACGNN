import torch


def calculate_edge_homo1(edge_index, labels):

    num_nodes = labels.size(0)
    homophily_levels = torch.zeros(num_nodes)

    for i in range(num_nodes):
        neighbors = edge_index[1, edge_index[0] == i]
        num_same_label_neighbors = (
            labels[neighbors] == labels[i]).sum().item()
        total_neighbors = neighbors.size(0)

        if total_neighbors > 0:
            homophily_levels[i] = num_same_label_neighbors / total_neighbors

    return round(homophily_levels.mean().item(), 2)


def calculate_edge_homo2(edge_index, labels):

    homophily_levels = (labels[edge_index[1]] == labels[edge_index[0]]).float()

    return round(homophily_levels.mean().item(), 2)


from utils.dataset import load_data


def neighbor_stat(dataset_name, run=0):
    import numpy as np
    edge_index, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = load_data(
        dataset_name, return_adj=False, run=run)

    num_nodes = features.shape[0]
    num_classes = len(np.unique(labels))
    degree = torch.zeros(num_nodes)
    for src, tgt in edge_index.t().tolist():
        degree[src] += 1
        degree[tgt] += 1
    for i in range(num_classes):
        print('class {} average degree: {}'.format(i,
                                                   degree[labels == i].mean()))

    neighbors = [[] for _ in range(features.shape[0])]
    neighbor_labels_distribution = torch.zeros(num_labels, num_labels)
    for src, tgt in edge_index.t().tolist():
        neighbors[src].append(tgt)

    neighbors = [neighbor for neighbor in neighbors if neighbor]

    class_counts = torch.zeros(labels.max() + 1)
    for node, neighbor_list in enumerate(neighbors):
        neighbor_labels = labels[neighbor_list]

        class_probs = torch.zeros(labels.max() + 1)
        for label in neighbor_labels:
            class_probs[label] += 1
        class_probs /= len(neighbor_list)
        neighbor_labels_distribution[labels[node]] += class_probs

    for class_id in range(num_labels):
        neighbor_labels_distribution[class_id] /= (labels == class_id).sum()
        print('class {} distribution: {}'.format(
            class_id,
            np.round(neighbor_labels_distribution[class_id].numpy(), 2)))


def get_graph_stat(dataset_name):

    edge_index, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = load_data(
        dataset_name, norm='none', return_adj=False, run=0)

    print('Labels: {}, Nodes: {}, Edges: {}, Features: {}'.format(
        num_labels, features.size(0), edge_index.size(1), num_features))
    print('edge homo: {}'.format(calculate_edge_homo1(edge_index, labels)))


for dataset in [
        'photo', 'computers', 'film', 'chameleon', 'squirrel', 'tolokers',
        'deezer-europe'
]:
    print(dataset)
    get_graph_stat(dataset)
    print('---')
