import sys
import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid, PPI, Amazon, Reddit, Coauthor, PPI, TUDataset
from webkb_data import WebKB
import pdb
import pickle as pkl
from scipy.sparse import coo_matrix
from torch_geometric.utils import is_undirected, to_undirected

import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp

def matching_labels_distribution(dataset):
    # Build graph
    adj = coo_matrix(
        (np.ones(dataset[0].num_edges),
        (dataset[0].edge_index[0].numpy(), dataset[0].edge_index[1].numpy())),
        shape=(dataset[0].num_nodes, dataset[0].num_nodes))
    G = nx.Graph(adj)

    hop_1_matching_percent = []
    hop_2_matching_percent = []
    hop_3_matching_percent = []
    for n in range(dataset.data.num_nodes):
        hop_1_neighbours = list(nx.ego_graph(G, n, 1).nodes())
        hop_2_neighbours = list(nx.ego_graph(G, n, 2).nodes())
        hop_3_neighbours = list(nx.ego_graph(G, n, 3).nodes())
        node_label = dataset[0].y[n]
        hop_1_labels = dataset[0].y[hop_1_neighbours]
        hop_2_labels = dataset[0].y[hop_2_neighbours]
        hop_3_labels = dataset[0].y[hop_3_neighbours]
        matching_1_labels = node_label == hop_1_labels
        matching_2_labels = node_label == hop_2_labels
        matching_3_labels = node_label == hop_3_labels
        hop_1_matching_percent.append(matching_1_labels.float().sum()/matching_1_labels.shape[0])
        hop_2_matching_percent.append(matching_2_labels.float().sum()/matching_2_labels.shape[0])
        hop_3_matching_percent.append(matching_3_labels.float().sum()/matching_3_labels.shape[0])

    return hop_1_matching_percent, hop_2_matching_percent, hop_3_matching_percent


def get_dataset(name, normalize_features=False, transform=None, edge_dropout=None, node_feature_dropout=None,
                dissimilar_t = 1, cuda=False, permute_masks=None, lcc=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name, split="full")
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(path, name, split="full")
    elif name in ['Reddit']:
        dataset = Reddit(path)
    elif name.lower() in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, name)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dataset.data.y = dataset.data.y.long()
    if not is_undirected(dataset.data.edge_index):
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    if dissimilar_t < 1 and not permute_masks:
        label_distributions = torch.tensor(matching_labels_distribution(dataset)).cpu()
        dissimilar_neighbhour_train_mask = dataset[0]['train_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dissimilar_neighbhour_val_mask = dataset[0]['val_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dissimilar_neighbhour_test_mask = dataset[0]['test_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dataset.data.train_mask = dissimilar_neighbhour_train_mask
        dataset.data.val_mask = dissimilar_neighbhour_val_mask
        dataset.data.test_mask = dissimilar_neighbhour_test_mask

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    if permute_masks is not None:
        label_distributions = torch.tensor(matching_labels_distribution(dataset)).cpu()
        dataset.data = permute_masks(dataset.data, dataset.num_classes, lcc_mask=lcc_mask,
                                     dissimilar_mask=(label_distributions[0] <= dissimilar_t))

    if cuda:
        dataset.data.to('cuda')

    return dataset

def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    ep = 1e-10
    r_inv = np.power(rowsum + ep, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(torch.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


def load_data(name, split=0):
    # train_mask, val_mask, test_mask: np.ndarray, [True/False] * node_number
    dataset = get_dataset(name, normalize_features=True)
    data = dataset[0]
    # pdb.set_trace()
    train_index = torch.where(data.train_mask[split])[0]

    adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges), (data.num_nodes, data.num_nodes))
    adj_train = adj.index_select(0, train_index).index_select(1, train_index)
    y_train = data.y[train_index]
    val_index = np.where(data.val_mask[split])[0]
    y_val = data.y[val_index]
    test_index = np.where(data.test_mask[split])[0]
    y_test = data.y[test_index]

    num_train = adj_train.shape[0]

    features = data.x
    train_features = features[train_index]

    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)

    if dataset == 'pubmed':
        norm_adj = 1*sp.diags(np.ones(norm_adj.shape[0])) + norm_adj
        norm_adj_train = 1*sp.diags(np.ones(num_train)) + norm_adj_train

    # change type to tensor
    # norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
    # features = torch.FloatTensor(features)
    # norm_adj_train = sparse_mx_to_torch_sparse_tensor(norm_adj_train)
    # train_features = torch.FloatTensor(train_features)
    # y_train = torch.LongTensor(y_train)
    # y_test = torch.LongTensor(y_test)
    # test_index = torch.LongTensor(test_index)

    return (norm_adj, features, norm_adj_train, train_features,
            y_train, y_val, y_test, train_index, val_index, test_index)


def get_batches(train_ind, train_labels, batch_size=64, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    pdb.set_trace()
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data('cora')
    pdb.set_trace()
