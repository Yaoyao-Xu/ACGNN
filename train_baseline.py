import time
import torch
import torch.optim as optim
import numpy as np
from utils.util import get_metric, cal_grad_norm
from torch_geometric.seed import seed_everything
from config import config_parser
from utils.dataset import load_data
import os
from utils.util import float_list2str
from datetime import datetime
import warnings
from torch.nn import functional as F
from utils.baseline_model import *

warnings.filterwarnings("ignore")
import wandb

wandb.login()

args = config_parser()

seed_everything(args.seed)

cuda_id = "cuda:" + str(args.device)
device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")


def process_step(model, optimizer, features, labels, adj, idx, train=False):
    if args.cal_time:
        start = datetime.now()

    loss_fc = torch.nn.CrossEntropyLoss()
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        # hidden, output, hidden_angle, hidden_norm, label_loss = model(
        #     features, adj, epoch if train else None)
        output = model(features, adj)
        output.to(device)
        # print(output[idx].shape)
        # loss = loss_fc(output[idx].view(-1), labels[idx].to(device))
        loss = loss_fc(output[idx], labels[idx].to(device))
        metric = get_metric(output[idx], labels[idx].to(device), args.use_auc)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
    if args.cal_time:
        end = datetime.now()
        running_time = (end - start).total_seconds() * 1000  # milliseconds
    else:
        running_time = None

    return loss.item(), metric, running_time


def train(i=0, config=None, use_wandb=False):
    dataset = args.dataset
    current_time = time.strftime("%d_%H_%M_%S", time.localtime(time.time()))
    ckpt_file = './pretrained/' + "{}_{}".format(args.dataset,
                                                 current_time) + '.pt'

    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = load_data(
        dataset,
        norm=args.adj_norm,
        mode=args.split_mode,
        return_adj=True,
        # return_adj=False,
        run=i)
    features = features.to(device)
    adj = adj.to(device)
    print('num_features', num_features, 'num_class', num_labels)
    metric_fc = None
    if use_wandb:
        wandb.init(config=config)
        config = wandb.config
    else:
        config = args

    # model = GCN(nfeat =features.shape[1],
    #             nlayers = 4,
    #             nhid = args.hidden,
    #             nclass = num_labels ,
    #             dropout = args.dropout).to(device)

    model = GAT(
        nfeat=features.shape[1],
        nhid=32,
        nlayers=2,  # two attention layers, 2 layer for dim transform
        nclass=num_labels,
        dropout=args.dropout,
        alpha=0.2,
        nheads=8,
        use_sparse=False).to(device)

    # ##########################################################################
    # model = CGNN(
    #     num_features=features.shape[1],
    #     num_layers=args.layer,
    #     num_hidden=args.hidden,
    #     num_class=num_labels,
    #     dropout=config.dropout,
    #     labels=labels,
    #     num_nodes=features.shape[0],
    #     use_norm=use_norm,
    #     device=device,
    #     train_mask=idx_train,
    #     distance_type=args.distance_type,
    #     normed_mlp=args.normed_mlp,
    #     init_theta=args.init_theta,
    #     asy=args.asy,
    #     att_method=args.att_method,
    #     activation=args.activation,
    #     use_extra_norm=args.use_extra_norm,
    # ).to(device)

    # #########################################################################
    # model  = GGCN(
    #              nfeat= features.shape[1],
    #              nlayers = args.layer,
    #              nhidden = args.hidden,
    #              nclass = num_labels,
    #              dropout = config.dropout,
    #              decay_rate=0,
    #              exponent=0,
    #              use_degree=True,
    #              use_sign=True,
    #              use_decay=False,
    #              use_sparse=False,
    #              scale_init=0.5,
    #              deg_intercept_init=0.5,
    #              use_bn=False,
    #              use_ln=False).to(device)

    ##########################################################################
    # model= H2GCN(
    #         feat_dim=features.shape[1],
    #         hidden_dim=args.hidden,
    #         class_dim=num_labels,
    #         k= 2,
    #         dropout= 0.1,
    #         use_relu = True
    # ).to(device)

    ######################Geom_GNN###############################################################
    # edge_index=adj.to_sparse().to_edge_index()
    # model=GeomGCN(x=feature,
    #               edge_index=edge_index, edge_relation, num_features=features.shape[1], num_hidden, num_classes, dropout, layer_num=2, device='cpu')
    # #######

    #####################GPR GNN###############################################################
    # model = GPRGNN(nfeat=features.shape[1],  ## return_adj=False
    #                nlayers = args.layer,
    #                nhidden = args.hidden,
    #                nclass = num_labels,
    #                dropout = config.dropout,
    #                dprate_GPRGNN = 0.7 ,
    #                Gamma_GPRGNN = None,
    #                alpha_GPRGNN = 1,           ## suqirrel is 0, chameleon is 1
    #                Init_GPRGNN = 'PPR',
    #                ppnp_GPRGNN = 'PPNP').to(device)

    ######################GCNII###############################################################
    # model = GCNII(nfeat = features.shape[1] ,
    #               nlayers = args.layer,
    #               nhidden =args.hidden ,
    #               nclass = num_labels,
    #               dropout = config.dropout ,
    #               lamda = 1.5,
    #               alpha = 0.2 ,
    #               variant = False).to(device)

    ######################MixHop###############################################################
    # model = MixHop(nfeat=features.shape[1],
    #                nlayers=args.layer,
    #                nhid=args.hidden,
    #                nclass=num_labels,
    #                dropout=config.dropout).to(device)

    ######################JKNet###############################################################
    # model = JKNet(nfeat = features.shape[1] ,
    #               nlayers = args.layer,
    #               nhid =args.hidden ,
    #               nclass = num_labels,
    #               mode='max').to(device)

    # model = DeepGCN(nfeat=features.shape[1],
    #                 nhid=args.hidden,
    #                 nclass=num_labels,
    #                 dropout=config.dropout,
    #                 nlayer=args.layer,
    #                 norm_mode="PN").to(device)

    if use_wandb:
        wandb.watch(model, log='all')

    metric_fc = None

    if metric_fc != None:
        optimizer = optim.Adam([{
            'params': model.params1,
            'weight_decay': 0
        }, {
            'params': model.params2,
            'weight_decay': config.weight_decay
        }, {
            'params': metric_fc.parameters(),
            'weight_decay': config.weight_decay
        }],
                               lr=config.lr)
    else:
        optimizer = optim.Adam(model.parameters(),
                               weight_decay=config.weight_decay,
                               lr=config.lr)
    wait = 0
    best_acc = 0
    best_model = model
    best_epoch = 0
    running_time = {'train': [], 'val': [], 'test': []}

    for epoch in range(args.epoch):

        loss_tra, metric_tra, time_tra = process_step(model, optimizer, features, labels, adj, idx_train, train=True) #yapf: disable
        loss_val, metric_val, time_val = process_step(model, optimizer, features, labels, adj, idx_val, train=False) #yapf: disable
        loss_test, metric_test, time_test = process_step(model, optimizer, features, labels, adj, idx_test, train=False) #yapf: disable

        grad_norm_display = cal_grad_norm(model)
        if args.cal_time:
            running_time['train'].append(time_tra)
            running_time['val'].append(time_val)
            running_time['test'].append(time_test)

        if use_wandb:
            wandb.log({
                'train_loss': loss_tra,
                'train_acc': metric_tra,
                'val_loss': loss_val,
                'val_acc': metric_val,
                'test_loss': loss_test,
                'test_acc': metric_test,
                'grad_norm': grad_norm_display
            })

        # if epoch > 200:
        if metric_val > best_acc:
            best_acc = metric_val
            torch.save(model.state_dict(), ckpt_file)
            # print('save path', ckpt_file)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait == args.patience:
                print('Early stopping!', 'Total epochs:{}'.format(epoch + 1))
                break

    if args.cal_time:
        # calculate per epoch running time, avg+std
        running_time['train'] = float_list2str(running_time['train'])
        running_time['val'] = float_list2str(running_time['val'])
        running_time['test'] = float_list2str(running_time['test'])
        print('running time is ', running_time)
        with open('running_time.txt', 'a') as f:
            f.write(args.dataset + ' ' + model.__class__.__name__ + ' ' +
                    str(running_time) + '\n')

    print('best epoch is ', best_epoch, 'best val acc is ', best_acc)
    best_model.load_state_dict(
        torch.load(ckpt_file))  #this will load the best model on val
    os.remove(ckpt_file)
    # # best_model = model
    loss, acc, time_ = process_step(model,optimizer, features, labels, adj, idx_test, train=False) #yapf: disable
    metric_test = acc
    print('test acc is ', metric_test)

    return metric_test


if __name__ == '__main__':
    # config = {
    #     'lr':  0.01,
    #     'weight_decay': 0.0001,
    #     'dropout': 0.4,
    # }
    # args.lr = config['lr']
    # args.weight_decay = config['weight_decay']
    # args.dropout = config['dropout']
    print(args)
    acc_list = []
    # 0~9
    for i in range(1):
        acc = train(i, config=args, use_wandb=False)
        print('acc is ', acc)
        acc_list.append(acc)

    acc_list = [tensor.to(torch.device('cpu')) for tensor in acc_list]
    acc_list = np.array(acc_list)
    print('mean acc is ',
          round(np.mean(acc_list), 4) * 100, 'std is ',
          round(np.std(acc_list), 4) * 100)
