import argparse
import time

import torch
from torch import tensor
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import GCN
from sampler import Sampler_FastGCN, Sampler_ASGCN
from utils import load_data, get_batches, accuracy
from utils import sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import f1_score


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Fast',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=10,
                        help='the train epochs between two test')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=729, help='Random seed.')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(train_ind, train_labels, batch_size, train_times):
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(
                batch_inds)
            optimizer.zero_grad()
            output = model(sampled_feats, sampled_adjs)
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def test(test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


def test(test_adj, val_feats, val_labels, test_feats, test_labels):
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_feats, val_adj)
        test_outputs = model(test_feats, test_adj)
    loss_val = loss_fn(val_outputs, val_labels)
    loss_test = loss_fn(test_outputs, test_labels)
    val_micro_f1 = f1_score(val_labels, val_outputs.max(1)[1].cpu(), average='micro')
    test_micro_f1 = f1_score(test_labels, test_outputs.max(1)[1].cpu(), average='micro')
    val_macro_f1 = f1_score(val_labels, val_outputs.max(1)[1].cpu(), average='macro')
    test_macro_f1 = f1_score(test_labels, test_outputs.max(1)[1].cpu(), average='macro')

    return {
        'val_loss': loss_val,
        'test_loss': loss_test,
        'val_micro_f1': val_micro_f1,
        'test_micro_f1': test_micro_f1,
        'val_macro_f1': val_macro_f1,
        'test_macro_f1': test_macro_f1,
    }


if __name__ == '__main__':
    val_losses, val_accs, test_accs, test_macro_f1s = [], [], [], []
    for split in range(10):
        # load data, set superpara and constant
        args = get_args()
        adj, features, adj_train, train_features, y_train, y_val, y_test, train_index, val_index, test_index = \
            load_data(args.dataset, split=split)

        layer_sizes = [128, 128]
        input_dim = features.shape[1]
        train_nums = adj_train.shape[0]
        test_gap = args.test_gap
        nclass = y_train.max().item() + 1

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        # set device
        if args.cuda:
            device = torch.device("cuda")
            print("use cuda")
        else:
            device = torch.device("cpu")

        # data for train and test
        features = torch.FloatTensor(features).to(device)
        train_features = torch.FloatTensor(train_features).to(device)
        y_train = y_train.to(device)

        val_adj = [adj, adj[val_index, :]]
        val_feats = features
        val_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                   for cur_adj in val_adj]
        val_labels = y_val.to(device)

        test_adj = [adj, adj[test_index, :]]
        test_feats = features
        test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                    for cur_adj in test_adj]
        test_labels = y_test.to(device)

        # init the sampler
        if args.model == 'Fast':
            sampler = Sampler_FastGCN(None, train_features, adj_train,
                                      input_dim=input_dim,
                                      layer_sizes=layer_sizes,
                                      device=device)
        elif args.model == 'AS':
            sampler = Sampler_ASGCN(None, train_features, adj_train,
                                    input_dim=input_dim,
                                    layer_sizes=layer_sizes,
                                    device=device)
        else:
            print("model name error, no model named {}".format(args.model))
            exit()

        for _ in range(args.runs):
            # init model, optimizer and loss function
            model = GCN(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=nclass,
                        dropout=args.dropout,
                        sampler=sampler).to(device)
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
            loss_fn = F.nll_loss
            # loss_fn = torch.nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            best_val_acc = float(0)
            eval_info_early_model = None
            bad_counter = 0

            # train and test
            for epochs in range(0, args.epochs // test_gap):
                train_loss, train_acc, train_time = train(np.arange(train_nums),
                                                          y_train,
                                                          args.batchsize,
                                                          test_gap)
                eval_info = test(test_adj, val_feats, val_labels, test_feats, test_labels)

                if eval_info['val_micro_f1'] > best_val_acc or eval_info['val_loss'] < best_val_loss:
                    if eval_info['val_micro_f1'] >= best_val_acc and eval_info['val_loss'] <= best_val_loss:
                        eval_info_early_model = eval_info
                        # torch.save(model.state_dict(), './best_{}_appnp.pkl'.format(dataset.name))
                    best_val_acc = np.max((best_val_acc, eval_info['val_micro_f1']))
                    best_val_loss = np.min((best_val_loss, eval_info['val_loss']))
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter == args.patience:
                        break

            val_losses.append(eval_info_early_model['val_loss'])
            val_accs.append(eval_info_early_model['val_micro_f1'])
            test_accs.append(eval_info_early_model['test_micro_f1'])
            test_macro_f1s.append(eval_info_early_model['test_macro_f1'])

    val_losses, val_accs, test_accs, test_macro_f1s = tensor(val_losses), tensor(val_accs), tensor(test_accs), tensor(
        test_macro_f1s)

    print(
        'Val Loss: {:.4f} ± {:.3f}, Val Accuracy: {:.3f} ± {:.3f}, Test Accuracy: {:.3f} ± {:.3f}, Macro-F1: {:.3f} ± {:.3f}'.format(
            val_losses.mean().item(),
            val_losses.std().item(),
            val_accs.mean().item(),
            val_accs.std().item(),
            test_accs.mean().item(),
            test_accs.std().item(),
            test_macro_f1s.mean().item(),
            test_macro_f1s.std().item()))
