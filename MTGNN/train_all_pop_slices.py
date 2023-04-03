import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random, os
from util import *
from trainer import Optim
from sequence_dataset import SequenceDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader 
from model import LSTMClassifier, AdversarialDiscriminator
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
parser.add_argument('--num_nodes',type=int,default=14,help='number of nodes/variables') # used to be 137 for solar
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=14,help='k') # used to be 20 by default
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=7,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=5,help='input sequence length') # used to be 24*7 for solar
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=200,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=60,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=30,help='step_size')
parser.add_argument('--adv', type=bool, default=True, help='whether to add adverserial loss')
parser.add_argument('--schedule_interval',type=int,default=1,help='scheduler interval')
parser.add_argument('--schedule_ratio',type=float,default=0.001,help='multiplicative factor of learning rate decay')
args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)
adv_E_delay_epochs = 0
adv_D_delay_epochs = 0
num_epoch_discriminator = 30
adv_weight = 1.3
rse_weight = 1
adv_weight_str = str(adv_weight).replace('.', 'dot')
rse_weight_str = str(rse_weight).replace('.', 'dot')
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = False
    
def plot_tsne(train_dataloader, val_dataloader, test_dataloader, model, epoch, name):
    # gtnet_features = FeatureExtractor(model, layers=["end_conv_1"])
    output_lst = []
    edema_lst = []
    input_lst = []
    user_label_lst = []
    num_data_lst = [0, 0, 0]
    for c, dataloader in enumerate([train_dataloader, val_dataloader, test_dataloader]):
        for batch_data in dataloader:
            X = batch_data['X'].to(device)
            Y = batch_data['Y'].to(device)
            user_label = batch_data['user_label']
            user_label_lst.append(user_label)
            feature_size = X.shape[-1]

            X = torch.unsqueeze(X,dim=1)
            X = X.transpose(2,3)
            edema_label = Y[:, -1].unsqueeze(-1).cpu().detach().numpy()

            edema_lst.append(edema_label)

            num_data_lst[c] += X.shape[0]
            
            with torch.no_grad():
                output = model(X, CLS=True)['cls_emb']
                # print(output.shape)
                output_lst.append(output.squeeze(-1).view(output.shape[0], -1).cpu().detach().numpy())
                # input_lst.append(X.squeeze(-1).reshape(X.shape[0], -1).cpu().detach().numpy())
    edema_lst = np.concatenate(edema_lst)
    all_embeddings = np.concatenate(output_lst, axis=0)
    all_user_label = np.concatenate(user_label_lst)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
    # tsne_results = tsne.fit_transform(pca_result)
    tsne_results = tsne.fit_transform(all_embeddings)
    df_tsne = pd.DataFrame(tsne_results, columns=["X", "Y"])

    df_tsne["User_labels"] = all_user_label
    df_tsne["User_labels"] = df_tsne["User_labels"].apply(lambda i: str(i))

    df_tsne["Training_set"] = [True] * num_data_lst[0] + [False] * (num_data_lst[1] + num_data_lst[2])
    markers_dict = {
        True: 'o',
        False: 'X',
    }
    df_tsne["Edema_label"] = edema_lst
    fig, axs = plt.subplots(ncols=2, figsize=(12, 12))
    # plt.figure(figsize=(16,16))
    axs[0].set(ylim=(-25, 25))
    axs[0].set(xlim=(-25, 25))
    axs[1].set(ylim=(-25, 25))
    axs[1].set(xlim=(-25, 25))
    sns.scatterplot(
        x="X", y="Y",
        hue="User_labels",
        style="Training_set",
        data=df_tsne,
        legend="full", s=70,
        alpha=0.9,
        markers=markers_dict,
        ax=axs[0]
    )
    sns.scatterplot(
        x="X", y="Y",
        hue="Edema_label",
        style="Training_set",
        data=df_tsne,
        legend="full", s=70,
        alpha=0.9,
        markers=markers_dict,
        ax=axs[1]
    )

    plt.savefig(os.path.join('/', 'mnt', 'results', 'plots', f'tsne_epoch{epoch}_adv{adv_weight_str}_l1{rse_weight_str}_{name}'))

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
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
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

    train_loss_E_class = 0
    adv_loss_E = 0
    train_adv_loss_D = 0
    n_samples = 0
    iter = 0
    train_adv_correct_num = 0
    for batch_data in dataloader:
        model.train()
        if args.adv:
            discriminator.train()
        X = batch_data['X'].to(device)
        Y = batch_data['Y'].to(device)
        user_label = batch_data['user_label'].to(device)
        feature_size = X.shape[-1]
        
        if model_type == 'GNN':
            optim_E.zero_grad()
            X = torch.unsqueeze(X,dim=1)
            X = X.transpose(2,3)
            if iter % args.step_size == 0:
                perm = np.random.permutation(range(args.num_nodes))
            # num_sub = int(args.num_nodes / args.num_split)
            # for j in range(args.num_split):
            #     if j != args.num_split - 1:
            #         id = perm[j * num_sub:(j + 1) * num_sub]
            #     else:
            #         id = perm[j * num_sub:]
            id = perm
            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :]
            ty = Y[:, :]
            output_dict = model(tx, CLS=args.adv)
            output = output_dict['output']
            if args.adv:
                cls_emb = output_dict['cls_emb']
            output = torch.squeeze(output)

            loss_CLS = criterion(output, ty)
            train_loss_E_class += loss_CLS.item()
            if args.adv and epoch >= adv_D_delay_epochs:
                if epoch % 5 == 0:
                    discriminator.apply(weight_reset)
                for i in range(num_epoch_discriminator):
                    discriminator.train()
                    optim_D.zero_grad()

                    output_dict = model(tx, CLS=args.adv)
                    pred = discriminator(output_dict["cls_emb"].detach())
                    loss_discriminator = criterion_adv(pred, user_label)
                    loss_discriminator.backward()
                    optim_D.step()
                    if i == num_epoch_discriminator - 1:
                        train_adv_loss_D += loss_discriminator.item()
                        pred_class = torch.squeeze(pred.max(1)[1])
                        train_adv_correct_num += torch.sum(pred_class==user_label)
                        n_samples += pred_class.shape[0]
            
            if args.adv and epoch >= adv_E_delay_epochs:

                output_dict = model(tx, CLS=args.adv)
                fake_idx = torch.randperm(user_label.nelement())
                fake_user_label = user_label.view(-1)[fake_idx].view(user_label.size())
                loss_adv_E = criterion_adv(discriminator(output_dict["cls_emb"]), fake_user_label)
            else:
                loss_adv_E = 0

            loss_total = rse_weight * loss_CLS + adv_weight * loss_adv_E
            loss_total.backward()
            optim_E.step()

        elif model_type == "LSTM":
            output = model(X)
            output = torch.squeeze(output)
            loss = criterion(output, Y)
            loss.backward()
            train_loss_E_class += loss.item()
            n_samples += (output.size(0) * feature_size)
            grad_norm = optim.step()



        # if iter%100==0:
        #     print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * feature_size)))
        # iter += 1
    loss_dict = {'training_loss': train_loss_E_class, 'adv_loss_D': train_adv_loss_D, 'train_acc_D': 0 if n_samples == 0 else train_adv_correct_num/n_samples}
    return loss_dict

def main(user_dict, curr_slice, name, feature_lst, model_type="GNN", task_name='edema_pred', print_feature_corr=True, model_path=None):
    
    train_df_lst = []
    val_df_lst = []
    test_df_lst = []

    train_dataset_lst = []
    val_dataset_lst = []
    test_dataset_lst = []

    list_users = []
    for u in user_dict:
        if curr_slice >= user_dict[u]: # not available for this user
            continue
        list_users.append(u)
        file_name = f'/mnt/results/{task_name}/user_{u}_{task_name}_hyperimpute_slice{curr_slice}.csv'
        curr_all_data = np.loadtxt(file_name, delimiter=',')[:, :-1] # exclude edema coarse label for now
        # print(u)
        num_all_data, _ = curr_all_data.shape
        # val_split_idx = int(num_all_data * 0.6)
        # test_split_idx = int(num_all_data * 0.8)
        val_split_idx = 15
        test_split_idx = 15

        # test is the same as val for now
        curr_train_data = curr_all_data[:val_split_idx, :].copy()
        curr_val_data = curr_all_data[val_split_idx:, :].copy()
        # curr_val_data = curr_all_data[val_split_idx:test_split_idx, :]
        curr_test_data = curr_all_data[test_split_idx:, :].copy()

        # print(curr_train_data.shape)
        # print(curr_val_data.shape)
        # print(curr_test_data.shape)

        train_df_lst.append(curr_train_data)
        val_df_lst.append(curr_val_data)
        test_df_lst.append(curr_test_data)

    # normalization

    # # over all population
    # normalized_train_df_lst, min_value_lst, max_value_lst = min_max_normalization(train_df_lst)
    # normalized_val_df_lst, _, _ = min_max_normalization(val_df_lst, min_value_lst=min_value_lst, max_value_lst=max_value_lst)
    # normalized_test_df_lst, _, _ = min_max_normalization(test_df_lst, min_value_lst=min_value_lst, max_value_lst=max_value_lst)

    # over each individual, use only first 2 week data
    normalized_train_df_lst = []
    normalized_val_df_lst = []
    normalized_test_df_lst = []
    for i in range(len(list_users)):
        curr_train_data = train_df_lst[i]
        curr_val_data = val_df_lst[i]
        curr_test_data = test_df_lst[i]

        first_two_week = curr_train_data[:15, :]
        rest = curr_train_data[15:, :]
        curr_train_data[:, -1] = curr_train_data[:, -1].astype(int)
        curr_val_data[:, -1] = curr_val_data[:, -1].astype(int)
        curr_test_data[:, -1] = curr_test_data[:, -1].astype(int)
        

        normalized_first_two_week, min_value_lst, max_value_lst = min_max_normalization([first_two_week])
        normalized_rest, _, _ = min_max_normalization([rest], min_value_lst=min_value_lst, max_value_lst=max_value_lst)

        normalized_val, _, _ = min_max_normalization([curr_val_data], min_value_lst=min_value_lst, max_value_lst=max_value_lst)
        normalized_test, _, _ = min_max_normalization([curr_test_data], min_value_lst=min_value_lst, max_value_lst=max_value_lst)

        normalized_train = np.concatenate((normalized_first_two_week, normalized_rest), axis=0)
        
        normalized_train_df_lst.append(normalized_train)
        # print(normalized_train[:, -1])
        
        if np.sum(np.isnan(normalized_train)) > 0:
            print(list_users[i])

        normalized_val_df_lst.append(normalized_val)
        
        normalized_test_df_lst.append(normalized_test)
    
    


    # create sequential datasets
    for count, curr_train_data in enumerate(normalized_train_df_lst):
        curr_train_dataset = SequenceDataset(curr_train_data, args.horizon, args.seq_in_len, device, user_id=count)
        train_dataset_lst.append(curr_train_dataset)
    for count, curr_val_data in enumerate(normalized_val_df_lst):
        curr_val_dataset = SequenceDataset(curr_val_data, args.horizon, args.seq_in_len, device, user_id=count)
        val_dataset_lst.append(curr_val_dataset)
    for count, curr_test_data in enumerate(normalized_test_df_lst):
        curr_test_dataset = SequenceDataset(curr_test_data, args.horizon, args.seq_in_len, device, user_id=count)
        test_dataset_lst.append(curr_test_dataset)
    
    # aggregate them
    aggregated_train_dataset = ConcatDataset(train_dataset_lst)
    aggregated_val_dataset = ConcatDataset(val_dataset_lst)
    aggregated_test_dataset = ConcatDataset(test_dataset_lst)

    train_dataloader = DataLoader(aggregated_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(aggregated_val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(aggregated_test_dataset, batch_size=args.batch_size, shuffle=False)
    # args.data = f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute.csv'
    args.save = f'/mnt/results/model/model_{name}_{model_type}_adv{args.adv}_slice{curr_slice}.pt'
    print(vars(args))

    if model_type == "GNN":
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
        if model_path is None:
            model.apply(weight_reset)
        else:
            stored_model = torch.load(model_path)
            model.load_state_dict(stored_model.state_dict())
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
    decoder = nn.Sequential(
        nn.Linear(in_features=args.end_channels * args.num_nodes, out_features=128, bias=True),
        nn.ReLU(),
        nn.LayerNorm(128),
        nn.Linear(in_features=128, out_features=6, bias=True)
    )
    if args.adv:
        lr_ADV = 0.001  # learning rate for discriminator, used when ADV is True
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
            train_acc_D = loss_dict['train_acc_D']
            # train_loss = train(train_dataloader, model, criterion, optim, device, model_type, epoch=epoch)
            val_loss, val_rae, val_corr, feat_corr_lst = evaluate(val_dataloader, model, evaluateL2, evaluateL1, device, model_type)
            # print('edema_corr', edema_corr)
            edema_corr = feat_corr_lst[-1]
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | adv loss {:5.4f} | adv acc {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, adv_loss, train_acc_D, val_loss, val_rae, val_corr, edema_corr), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            feature_corr_output_str = ''
            if print_feature_corr:
                for idx, feat in enumerate(feature_lst):
                    feature_corr_output_str += f'{feat}_corr {feat_corr_lst[idx]} | '
                print(feature_corr_output_str)

            # if best_val < edema_corr:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val = edema_corr
            if epoch % 5 == 0:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
            if epoch % 20 == 0:
                plot_tsne(train_dataloader, val_dataloader, test_dataloader, model, epoch, name)
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
    test_acc, test_rae, test_corr, tfeat_corr_lst = evaluate(test_dataloader, model, evaluateL2, evaluateL1, device, model_type)
    return vtest_acc, vtest_rae, vtest_corr, vfeat_corr_lst, test_acc, test_rae, test_corr, tfeat_corr_lst

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
    feature_name_lst = [#'Active calories',
    'Calories',
    'Daily movement','Minutes of high-intensity activity','Minutes of inactive',
    'Minutes of low-intensity activity','Minutes of medium-intensity activity',
    # 'High-intensity MET',
    # 'Inactive MET',
    # 'Low-intensity MET',
    # 'Medium-intensity MET',
    'Minutes of non-wear','Minutes of rest','Total daily steps','Impedance magnitude','Impedance phase',
    'Weight','Respiratory rate','Edema']
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    feat_corr_lst = []
    test_feat_corr_lst = []
    num_runs = 1
    user_dict = {604: 3, 186: 4, 2267: 3, 2109: 4, 173: 4, 410: 1, 95: 3, 1976: 2, 28: 4, 168: 4, 1426: 
    4, 2001: 3, 55: 3, 2142: 3, 1389: 3, 1714: 4, 118: 4, 2102: 2, 2066: 3, 581: 4, 38: 2, 1400: 1, 
    983: 2, 137: 2, 74: 1, 64: 4, 404: 2, 2151: 1, 1717: 2, 159: 4, 1708: 4, 2159: 4, 1021: 2, 1989: 4, 
    135: 3, 991: 2, 2015: 2, 428: 2, 193: 3, 1747: 3, 1985: 2, 969: 2, 2169: 3, 174: 2, 980: 4, 2056: 4, 
    66: 1, 290: 4, 1744: 3, 1373: 1, 1709: 3, 1658: 3, 2083: 3, 2174: 2, 2134: 2, 2068: 1, 2080: 3, 1716: 4, 
    30: 2, 1745: 1, 2061: 3, 977: 3, 2113: 3, 2041: 3, 966: 1, 2176: 2, 987: 2, 1429: 4, 1696: 4, 429: 1, 185: 3, 
    2126: 2, 1038: 1, 1724: 4, 2065: 3, 293: 1, 39: 1, 1427: 2, 234: 1, 53: 4, 603: 1, 1728: 2, 1988: 1, 1367: 1, 
    1757: 3, 2038: 2, 47: 1, 192: 3, 992: 3, 1715: 1, 2100: 1, 989: 2, 
    2032: 2, 407: 2, 1440: 1, 2160: 1, 190: 2, 2058: 1, 1750: 1, 1436: 1, 1393: 1, 1000: 1, 1431: 1, 289: 1}
    curr_slice = 3
    if curr_slice != 0:
        previous_model = os.path.join('/', 'mnt', 'results', 'model', f'model_all_feat_0_advweight_1dot3_indinorm_slices{curr_slice-1}_GNN_advTrue_slice{curr_slice-1}.pt')
    else:
        previous_model = None
    # 42
    # torch.manual_seed(2139)
    seed_everything(2139)
    for i in range(num_runs):
        val_acc, val_rae, val_corr, vfeat_corr, test_acc, test_rae, test_corr, test_feat_corr = main(user_dict, curr_slice, f'all_feat_{i}_advweight_{adv_weight_str}_indinorm_slices{curr_slice}', 
        feature_lst=feature_name_lst, model_type='GNN', task_name="edema_pred_window", model_path=previous_model)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        feat_corr_lst.append(vfeat_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        test_feat_corr_lst.append(test_feat_corr)
    print('\n\n')
    print(f'{num_runs} runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))


    # feat_str = 'valid\t'
    # mean_str = 'mean\t'
    # std_str = 'std\t'
    # for feat in feature_name_lst:
    #     feat_str += feat + '\t'
    # for i in range(mean_feat_corr_lst.shape[0]):
    #     mean_str += ("{:5.4f}\t".format(mean_feat_corr_lst[i]))
    #     std_str += ("{:5.4f}\t".format(std_feat_corr_lst[i]))

    feat_corr_lst = np.vstack(feat_corr_lst)
    mean_feat_corr_lst = np.mean(feat_corr_lst, axis=0)
    std_feat_corr_lst = np.std(feat_corr_lst, axis=0)
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
    print('-'*10, 'test', '-' * 10)
    feat_corr_lst = np.vstack(test_feat_corr_lst)
    mean_feat_corr_lst = np.mean(test_feat_corr_lst, axis=0)
    std_feat_corr_lst = np.std(test_feat_corr_lst, axis=0)
    feat_str = 'test&'
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