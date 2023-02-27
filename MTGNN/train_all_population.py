import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *
from trainer import Optim
from sequence_dataset import SequenceDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader 
def evaluate(dataloader, model, evaluateL2, evaluateL1, device):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)
        feature_size = X.shape[-1]
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
        scale = np.ones(feature_size)
        scale = torch.from_numpy(scale).float().to(device)
        scale = scale.expand(output.size(0), feature_size)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * feature_size)

    rse = math.sqrt(total_loss / n_samples) # / data.rse
    rae = (total_loss_l1 / n_samples) # / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation


def train(dataloader, model, criterion, optim, device):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)
        feature_size = X.shape[-1]
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx,id)
            output = torch.squeeze(output)
            scale = np.ones(feature_size)
            scale = torch.from_numpy(scale).float().to(device)
            scale = scale.expand(output.size(0), feature_size)
            scale = scale[:,id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * feature_size)
            grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * feature_size)))
        iter += 1
    return total_loss / n_samples

selected_user = 1431
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default=f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=f'/mnt/results/model/model_{selected_user}.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=0) # already normalized
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=19,help='number of nodes/variables') # used to be 137 for solar
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=19,help='k') # used to be 20 by default
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=14,help='input sequence length') # used to be 24*7 for solar
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=7)
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=50,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def main(list_users, name):
    torch.manual_seed(42)
    train_dataset_lst = []
    val_dataset_lst = []
    for u in list_users:
        file_name = f'/mnt/results/user_{u}_activity_bodyport_hyperimpute_normalized.csv'
        curr_all_data = np.loadtxt(file_name, delimiter=',')
        num_all_data, _ = curr_all_data.shape
        curr_train_data = curr_all_data[:int(round(num_all_data * 0.8)), :]
        curr_val_data = curr_all_data[int(round(num_all_data * 0.8)):, :]
        curr_train_dataset = SequenceDataset(curr_train_data, args.horizon, args.seq_in_len, device)
        curr_val_dataset = SequenceDataset(curr_val_data, args.horizon, args.seq_in_len, device)
        train_dataset_lst.append(curr_train_dataset)
        val_dataset_lst.append(curr_val_dataset)
    aggregated_train_dataset = ConcatDataset(train_dataset_lst)
    aggregated_val_dataset = ConcatDataset(val_dataset_lst)

    train_dataloader = DataLoader(aggregated_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(aggregated_val_dataset, batch_size=args.batch_size, shuffle=False)
    # args.data = f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute.csv'
    args.save = f'/mnt/results/model/model_{name}.pt'
    print(vars(args))
    # Data = DataLoaderS(args.data, 0.8, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    model = model.to(device)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)


    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(train_dataloader, model, criterion, optim, device)
            val_loss, val_rae, val_corr = evaluate(val_dataloader, model, evaluateL2, evaluateL1, device)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            if val_loss < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
            # if epoch % 5 == 0:
            #     test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
            #                                          args.batch_size)
            #     print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    vtest_acc, vtest_rae, vtest_corr = evaluate(val_dataloader, model, evaluateL2, evaluateL1, device)
    # test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
    #                                      args.batch_size)
    # print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    test_acc, test_rae, test_corr = None, None, None
    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr

if __name__ == "__main__":
    list_users_above_criteria = [
        1032,
        581,
        407,
        290,
        1436,
        1000,
        95,
        1386,
        1431,
        992,
        1717,
        1441,
        122,
        977,
        293,
        1700,
        1744,
        622,

        192,
        1373,
        84,
        1393,
        1432,
        1378,
        225,
        1753,
        2084,
        969,
        280,
        99,
        53,
        983,
        2068,
        193,
        2056,
        2016,
        2109, 
        1995,
        1706,
        2015,
        186,
        137,
        1658,
        2083,
        1383,
        429,
        279]
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    num_runs = 1
    for i in range(num_runs):
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main(list_users_above_criteria, 'all_pop')
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print(f'{num_runs} runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    # not_available = []
    # for user in list_users_above_criteria:
    #     try:
    #         for i in range(num_runs):
    #             val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main(user)
    #             vacc.append(val_acc)
    #             vrae.append(val_rae)
    #             vcorr.append(val_corr)
    #             acc.append(test_acc)
    #             rae.append(test_rae)
    #             corr.append(test_corr)
    #         print('\n\n')
    #         print(f'{num_runs} runs average')
    #         print('\n\n')
    #         print("valid\trse\trae\tcorr")
    #         print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    #         print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    #     except:
    #         not_available.append(user)
    # print("not available", not_available)
    # print('\n\n')
    # print("test\trse\trae\tcorr")
    # print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    # print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))
