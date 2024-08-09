import time
import torch
import torch.optim as optim
from utils.model import CGNN
from utils.util import get_metric, cal_grad_norm
from torch_geometric.seed import seed_everything
from config import config_parser
from utils.dataset import load_data
from utils.util import float_list2str
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
import wandb

wandb.login()

args = config_parser()

seed_everything(args.seed)

device = torch.device("cuda:" +
                      str(args.device) if torch.cuda.is_available() else "cpu")


def process_step(model, optimizer, features, labels, adj, idx, train=False):

    loss_fc = torch.nn.CrossEntropyLoss()

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    if args.cal_time:
        start = datetime.now()

    with torch.set_grad_enabled(train):
        hidden, output, hidden_angle, hidden_norm, separate_loss, supervised_loss = model(
            features, adj)

        hidden_norm = torch.abs(output)
        loss_norm = loss_fc(hidden_norm[idx], labels[idx].to(device))
        loss = loss_norm
        output = hidden_norm

        if train and args.separate_loss:
            loss += separate_loss
        if args.supervised_loss:
            loss += supervised_loss

        metric = get_metric(output[idx], labels[idx], args.use_auc)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

    if args.cal_time:
        end = datetime.now()
        running_time = (end - start).total_seconds() * 1000
    else:
        running_time = None

    return loss.item(), metric, (separate_loss, supervised_loss), running_time


def train(i=0, config=None, use_wandb=True):
    dataset = args.dataset
    current_time = time.strftime("%d_%H_%M_%S", time.localtime(time.time()))
    if os.path.exists('./pretrained') == False:
        os.makedirs('./pretrained')
    ckpt_file = './pretrained/' + "{}_{}".format(args.dataset,
                                                 current_time) + '.pt'

    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = load_data(
        dataset,
        norm=args.adj_norm,
        mode=args.split_mode,
        return_adj=True,
        run=i)
    features = features.to(device)
    adj = adj.to(device)
    print('num_features', num_features, 'num_class', num_labels)
    features = features.type(torch.complex64)

    if use_wandb:
        wandb.init(config=config)
        config = wandb.config
    else:
        config = args

    model = CGNN(
        num_features=features.shape[1],
        num_layers=args.layer,
        num_hidden=args.hidden,
        num_class=num_labels,
        dropout=config.dropout,
        labels=labels,
        num_nodes=features.shape[0],
        device=device,
        train_mask=idx_train,
        sr_weight=args.sr_weight,
        extra_dim=args.extra_dim,
        select=args.select,
        activation=args.activation,
        reciprocal=args.reciprocal,
    ).to(device)

    if use_wandb:
        wandb.watch(model, log='all')

    optimizer = optim.Adam(model.parameters(),
                           weight_decay=config.weight_decay,
                           lr=config.lr)
    wait = 0
    best_acc = 0
    best_model = model
    best_epoch = 0
    test_acc_from_best_val_model = 0
    running_time = {'train': [], 'val': [], 'test': []}

    for epoch in range(args.epoch):
        loss_tra, metric_tra, losses_tra, time_tra = process_step(model,optimizer, features, labels, adj, idx_train, train=True) #yapf: disable
        loss_val, metric_val, losses_val, time_val = process_step(model,optimizer, features, labels, adj, idx_val, train=False) #yapf: disable
        loss_test, metric_test, losses_test, time_test = process_step(model, optimizer, features, labels, adj, idx_test, train=False) #yapf: disable

        if args.cal_time:
            running_time['train'].append(time_tra)
            running_time['val'].append(time_val)
            running_time['test'].append(time_test)

        grad_norm_display = cal_grad_norm(model)
        if metric_val > best_acc:
            best_acc = metric_val
            test_acc_from_best_val_model = metric_test
            torch.save(model.state_dict(), ckpt_file)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait == args.patience:
                print('Early stopping!', 'Total epochs:{}'.format(epoch + 1))
                break

        if use_wandb:
            wandb.log({ #yapf: disable
                'train_loss': loss_tra,
                'train_acc': metric_tra,
                'val_loss': loss_val,
                'val_acc': metric_val,
                'test_loss': loss_test,
                'test_acc': metric_test,
                'grad_norm': grad_norm_display,
                'train_separate_loss': losses_tra[0],
                'train_supervised_loss': losses_tra[1],
                'val_separate_loss': losses_val[0],
                'val_supervised_loss': losses_val[1],
                'test_separate_loss': losses_test[0],
                'test_supervised_loss': losses_test[1],
                'test_acc_from_best_val_model': test_acc_from_best_val_model,
            })

    if args.cal_time:

        running_time['train'] = float_list2str(running_time['train'])
        running_time['val'] = float_list2str(running_time['val'])
        running_time['test'] = float_list2str(running_time['test'])
        print('running time is ', running_time)
        with open('running_time.txt', 'a') as f:
            f.write(args.dataset + ' ' + str(running_time) + '\n')

    print('best epoch is ', best_epoch, 'best val acc is ',
          round(best_acc.item(), 4))
    best_model.load_state_dict(torch.load(ckpt_file))
    os.remove(ckpt_file)
    loss, acc, losses, time_ = process_step(model,optimizer, features, labels, adj, idx_test, train=False) #yapf: disable
    final_test_acc = acc.item()
    print('test acc is ', round(final_test_acc, 4))

    return final_test_acc


if __name__ == '__main__':
    if not args.search_hyper_mode:
        acc_list = []
        print('args is ', args)
        for i in range(10):
            acc = train(i, config=args, use_wandb=False)
            acc_list.append(acc)
        print('mean acc is ', float_list2str(acc_list))
    else:
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'test_acc_from_best_val_model',
                'goal': 'maximize'
            },
            'parameters': {
                'lr': {
                    'distribution': 'uniform',
                    'min': 0.0001,
                    'max': 0.05,
                },
                'weight_decay': {
                    'distribution': 'uniform',
                    'min': 0.0,
                    'max': 0.01,
                },
                'dropout': {
                    'distribution': 'uniform',
                    'min': 0.0,
                    'max': 0.9,
                },
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 50
            }
        }
        sweep_id = wandb.sweep(sweep_config, project=args.sweep_project_name)
        wandb.agent(sweep_id, function=train, count=1000)
