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
from model import LSTMClassifier, AdversarialDiscriminator


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')

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
parser.add_argument('--end_channels',type=int,default=7,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=7,help='input sequence length') # used to be 24*7 for solar
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=128,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=70,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=30,help='step_size')
parser.add_argument('--adv', type=bool, default=True, help='whether to add adverserial loss')
parser.add_argument('--schedule_interval',type=int,default=1,help='scheduler interval')
parser.add_argument('--schedule_ratio',type=float,default=0.001,help='multiplicative factor of learning rate decay')
args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)
adv_E_delay_epochs = 4
adv_D_delay_epochs = 3
num_epoch_discriminator = 50
adv_weight = 1.7
def evaluate(dataloader, model, evaluateL2, evaluateL1, device, model_type):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for batch_data in dataloader:
        X = batch_data['X'].to(device)
        Y = batch_data['Y'].to(device)
        user_label = batch_data['user_label'].to(device)
        feature_size = X.shape[-1]
        if model_type == 'GNN':
            X = torch.unsqueeze(X,dim=1)
            X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)['output']
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

    rse = math.sqrt(total_loss) # / data.rse
    rae = (total_loss_l1) # / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    def get_correlation(predict, Ytest):
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
        return correlation
    correlation = get_correlation(predict, Ytest)

    edema_correlation = get_correlation(predict[:, -1], Ytest[:, -1])
    corr_lst = []

    for feat_idx in range(predict.shape[1]):
        corr_lst.append(get_correlation(predict[:, feat_idx], Ytest[:, feat_idx]))
    
    return rse, rae, correlation, corr_lst


def train(dataloader, model, criterion, optim, device, model_type, epoch, optim_D=None, optim_E=None, discriminator=None, criterion_adv=None):
    model.train()
    if args.adv:
        discriminator.train()
    total_loss = 0
    adv_loss_E = 0
    adv_loss_D = 0
    n_samples = 0
    iter = 0
    for batch_data in dataloader:
        X = batch_data['X'].to(device)
        Y = batch_data['Y'].to(device)
        user_label = batch_data['user_label'].to(device)
        feature_size = X.shape[-1]
        model.zero_grad()
        if model_type == 'GNN':
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
                
                output_dict = model(tx, id, CLS=args.adv)
                output = output_dict['output']
                if args.adv:
                    cls_emb = output_dict['cls_emb']

                output = torch.squeeze(output)
                scale = np.ones(feature_size)
                scale = torch.from_numpy(scale).float().to(device)
                scale = scale.expand(output.size(0), feature_size)
                scale = scale[:,id]
                if args.adv and epoch > adv_E_delay_epochs:
                    discriminator.zero_grad()
                    loss_adv_E = -criterion_adv(discriminator(output_dict["cls_emb"]), user_label)
                else:
                    loss_adv_E = 0
                loss = criterion(output * scale, ty * scale) + adv_weight * loss_adv_E
                loss.backward()
                total_loss += loss.item()
                n_samples += (output.size(0) * feature_size)
                grad_norm = optim.step()
        elif model_type == "LSTM":
            output = model(X)
            output = torch.squeeze(output)
            loss = criterion(output, Y)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * feature_size)
            grad_norm = optim.step()
    # TRAIN DISCRIMINATOR
    if model_type == "GNN" and args.adv and epoch > adv_D_delay_epochs:
        for i in range(num_epoch_discriminator):
            for batch_data in dataloader:
                X = batch_data['X'].to(device)
                Y = batch_data['Y'].to(device)
                user_label = batch_data['user_label'].to(device)
                model.zero_grad()
                discriminator.zero_grad()
                feature_size = X.shape[-1]
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
                    output_dict = model(tx, id, CLS=args.adv)
                    loss_adv_D = criterion_adv(
                    discriminator(output_dict["cls_emb"].detach()), user_label)
                    
                    if i==num_epoch_discriminator-1:
                        adv_loss_D += loss_adv_D.item()
                    loss_adv_D.backward()
                    optim_D.step()

                    # # TRAINING ENCODER
                    # model.zero_grad()
                    # discriminator.zero_grad()
                    # loss_adv_E = -criterion_adv(
                    #     discriminator(output_dict["cls_emb"]), user_label)
                    # if epoch > adv_E_delay_epochs:
                    #     # model.zero_grad()
                    #     discriminator.zero_grad()
                    #     loss_adv_E.backward()
                    #     optim_E.step()


        # if iter%100==0:
        #     print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * feature_size)))
        # iter += 1
    loss_dict = {'training_loss': total_loss, 'adv_loss_D': adv_loss_D}
    return loss_dict

def main(list_users, name, feature_lst, model_type="GNN", task_name='edema_pred', print_feature_corr=True):
    
    train_df_lst = []
    val_df_lst = []

    train_dataset_lst = []
    val_dataset_lst = []

    for u in list_users:
        file_name = f'/mnt/results/{task_name}/user_{u}_{task_name}_hyperimpute.csv'
        curr_all_data = np.loadtxt(file_name, delimiter=',')
        print(u)
        num_all_data, _ = curr_all_data.shape
        curr_train_data = curr_all_data[:int(round(num_all_data * 0.8)), :]
        curr_val_data = curr_all_data[int(round(num_all_data * 0.8)):, :]
        train_df_lst.append(curr_train_data)
        val_df_lst.append(curr_val_data)

    # normalization
    normalized_train_df_lst, min_value_lst, max_value_lst = min_max_normalization(train_df_lst)
    normalized_val_df_lst, _, _ = min_max_normalization(val_df_lst, min_value_lst=min_value_lst, max_value_lst=max_value_lst)

    # create sequential datasets
    for count, curr_train_data in enumerate(normalized_train_df_lst):
        curr_train_dataset = SequenceDataset(curr_train_data, args.horizon, args.seq_in_len, device, user_id=count)
        train_dataset_lst.append(curr_train_dataset)
    for count, curr_val_data in enumerate(normalized_val_df_lst):
        curr_val_dataset = SequenceDataset(curr_val_data, args.horizon, args.seq_in_len, device, user_id=count)
        val_dataset_lst.append(curr_val_dataset)
    
    # aggregate them
    aggregated_train_dataset = ConcatDataset(train_dataset_lst)
    aggregated_val_dataset = ConcatDataset(val_dataset_lst)

    train_dataloader = DataLoader(aggregated_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(aggregated_val_dataset, batch_size=args.batch_size, shuffle=False)
    # args.data = f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute.csv'
    args.save = f'/mnt/results/model/model_{name}_{model_type}_adv{args.adv}.pt'
    print(vars(args))

    if model_type == "GNN":
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    elif model_type == "LSTM":
        model = LSTMClassifier(feature_size=args.num_nodes, n_state=args.num_nodes, hidden_size=30, rnn='LSTM')
    model = model.to(device)

    print(args)
    if model_type == 'GNN':
        print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    optimizer_D = None
    optimizer_E = None
    discriminator = None
    criterion_adv = None
    if args.adv:
        lr_ADV = 0.005  # learning rate for discriminator, used when ADV is True
        discriminator = AdversarialDiscriminator(
            d_model=args.end_channels * args.num_nodes,
            n_cls=len(list_users)).to(device)

        criterion_adv = nn.CrossEntropyLoss().to(device)  # consider using label smoothing
        optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, args.schedule_interval, gamma=args.schedule_ratio)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, args.schedule_interval, gamma=args.schedule_ratio)

    if args.L1Loss:
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)
    evaluateL2 = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss().to(device)


    best_val = 0
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            loss_dict = train(train_dataloader, model, criterion, optim, device, model_type, epoch=epoch, optim_D=optimizer_D, optim_E=optimizer_E, 
                discriminator=discriminator, criterion_adv=criterion_adv)
            train_loss = loss_dict['training_loss']
            adv_loss = loss_dict['adv_loss_D']
            # train_loss = train(train_dataloader, model, criterion, optim, device, model_type, epoch=epoch)
            val_loss, val_rae, val_corr, feat_corr_lst = evaluate(val_dataloader, model, evaluateL2, evaluateL1, device, model_type)
            # print('edema_corr', edema_corr)
            edema_corr = feat_corr_lst[-1]
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | adv loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, adv_loss, val_loss, val_rae, val_corr, edema_corr), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            feature_corr_output_str = ''
            if print_feature_corr:
                for idx, feat in enumerate(feature_lst):
                    feature_corr_output_str += f'{feat}_corr {feat_corr_lst[idx]} | '
                print(feature_corr_output_str)

            if best_val < edema_corr:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = edema_corr

            # if epoch % 5 == 0:
            #     test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
            #                                          args.batch_size)
            #     print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)
            # if args.adv:
            #     scheduler_D.step()
            #     scheduler_E.step()
        with open(args.save, 'wb') as f:
            torch.save(model, f)



    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    vtest_acc, vtest_rae, vtest_corr, vfeat_corr_lst = evaluate(val_dataloader, model, evaluateL2, evaluateL1, device, model_type)
    # test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
    #                                      args.batch_size)
    # print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    test_acc, test_rae, test_corr = None, None, None
    return vtest_acc, vtest_rae, vtest_corr, vfeat_corr_lst, test_acc, test_rae, test_corr

if __name__ == "__main__":
    ## for emesis
    # list_users_above_criteria = [
    #     1032,
    #     581,
    #     407,
    #     290,
    #     1436,
    #     1000,
    #     95,
    #     1386,
    #     1431,
    #     992,
    #     1717,
    #     1441,
    #     122,
    #     977,
    #     293,
    #     1700,
    #     1744,
    #     622,

    #     192,
    #     1373,
    #     84,
    #     1393,
    #     1432,
    #     1378,
    #     225,
    #     1753,
    #     2084,
    #     969,
    #     280,
    #     99,
    #     53,
    #     983,
    #     2068,
    #     193,
    #     2056,
    #     2016,
    #     2109, 
    #     1995,
    #     1706,
    #     2015,
    #     186,
    #     137,
    #     1658,
    #     2083,
    #     1383,
    #     429,
    #     279]
    # for edema
    list_users_above_criteria = [581, 407, 290, 1436, 1000, 95, 992, 1717, 293, 622, 291, 192, 1373, 225, 969, 280, 53, 983, 193, 186, 137, 
    #1383, 
    429]
    list_users_above_criteria = [28, 30, 38, 40, 42, 53, 54, 55, 64, 66, 67, 68, 74, 94, 95, 118, 135, 137, 159, 1373, 1000, 174, 186, 190, 192, 193, 
    # 1021, 
    # 976, 
    972, 225, 1004, 1429, 234, 280, 290, 291, 293, 404, 407, 408, 410, 1047, 428, 429, 980, 581, 603, 604, 622, 734, 983, 966, 969, 985, 987, 989, 991, 992, 997, 1024, 1041, 1403, 1038, 1367, 
    # 1383, 
    1389, 1422, 1426, 1427, 1436, 1440, 1444, 1453, 1717]

    list_users_above_criteria = [53, 55, 137, 159, 410, 581, 622, 987, 1426]
    feature_name_lst = ['Active calories','Calories','Daily movement','Minutes of high-intensity activity','Minutes of inactive',
    'Minutes of low-intensity activity','Minutes of medium-intensity activity','High-intensity MET','Inactive MET','Low-intensity MET',
    'Medium-intensity MET','Minutes of non-wear','Minutes of rest','Total daily steps','Impedance magnitude','Impedance phase',
    'Weight','Respiratory rate','Edema']
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    feat_corr_lst = []
    num_runs = 1
    torch.manual_seed(42)
    for i in range(num_runs):
        val_acc, val_rae, val_corr, vfeat_corr_lst, test_acc, test_rae, test_corr = main(list_users_above_criteria, f'all_pop_edema_{i}_advweight_{adv_weight}', feature_lst=feature_name_lst, model_type='GNN')
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        feat_corr_lst.append(vfeat_corr_lst)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print(f'{num_runs} runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    feat_corr_lst = np.vstack(feat_corr_lst)
    mean_feat_corr_lst = np.mean(feat_corr_lst, axis=0)
    std_feat_corr_lst = np.std(feat_corr_lst, axis=0)

    # feat_str = 'valid\t'
    # mean_str = 'mean\t'
    # std_str = 'std\t'
    # for feat in feature_name_lst:
    #     feat_str += feat + '\t'
    # for i in range(mean_feat_corr_lst.shape[0]):
    #     mean_str += ("{:5.4f}\t".format(mean_feat_corr_lst[i]))
    #     std_str += ("{:5.4f}\t".format(std_feat_corr_lst[i]))


    feat_str = 'valid&'
    mean_str = 'mean&'
    std_str = 'std&'
    for feat in feature_name_lst:
        feat_str += feat + '&'
    for i in range(mean_feat_corr_lst.shape[0]):
        mean_str += ("${:5.4f}\pm{:5.4f}$&".format(mean_feat_corr_lst[i], std_feat_corr_lst[i]))
        # std_str += ("{:5.4f}&".format(std_feat_corr_lst[i]))
    print('\n\n')
    print(feat_str)
    print(mean_str)
    # print(std_str)