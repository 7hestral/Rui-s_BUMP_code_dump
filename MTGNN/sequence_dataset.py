import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
from torch.utils.data import Dataset

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class SequenceDataset(Dataset):
    def __init__(self, data, horizon, window, device, normalize=0):
        self.horizon = horizon
        self.window = window
        self.raw_data = data
        self.data = None
        self._normalized(normalize)

        # self.scale = np.ones(self.feature_size)
        # self.scale = torch.from_numpy(self.scale).float()
        # self.scale = Variable(self.scale)

        self.device = device
        self.num_rows, self.feature_size = self.data.shape
        idx_set = range(self.window + self.horizon - 1, self.num_rows)
        self.data_X, self.data_Y = self._batchify(idx_set)
    
    def __len__(self):
        return(len(self.data_X))

    def __getitem__(self, idx):
        return self.data_X[idx, :, :], self.data_Y[idx, :]

    def _normalized(self, normalize):

        # normalized by the maximum value of entire matrix.
        if (normalize == 0):
            self.data = self.raw_data

        if (normalize == 1):
            self.data = self.raw_data / np.max(self.raw_data)

        # normlized by the maximum value of each row(sensor). Rui: should be columns here???
        if (normalize == 2):
            for i in range(self.feature_size):
                self.scale[i] = np.max(np.abs(self.raw_data[:, i]))
                self.data[:, i] = self.raw_data[:, i] / np.max(np.abs(self.raw_data[:, i]))

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.window, self.feature_size))
        Y = torch.zeros((n, self.feature_size))
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])
            Y[i, :] = torch.from_numpy(self.data[idx_set[i], :])
        return [X, Y]
    
class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.if_test = True
        if 1 - train - valid < 0.05:
            self.if_test = False
        self.window = window
        self.horizon = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.num_rows, self.feature_size = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.feature_size)
        print(self.scale)
        self._normalized(normalize)
        self._split(int(train * self.num_rows), int((train + valid) * self.num_rows), self.num_rows)

        self.scale = torch.from_numpy(self.scale).float()
        if self.if_test:
            tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.feature_size)
        else:
            tmp = self.valid[1] * self.scale.expand(self.valid[1].size(0), self.feature_size)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rse = normal_std(self.valid[1])
        print(tmp.shape)
        print(self.valid[1].shape)
        print(self.scale)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor). Rui: should be columns here???
        if (normalize == 2):
            for i in range(self.feature_size):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.window + self.horizon - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.num_rows)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        if self.if_test:
            self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.window, self.feature_size))
        Y = torch.zeros((n, self.feature_size))
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            # print(X.shape)
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


if __name__ == "__main__":
    selected_user = 74
    device = 'cuda:0'
    Data = DataLoaderS(f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute.csv', 0.8, 0.2, device, horizon=7, window=14, normalize=2)
    print(Data.rse)

    

            