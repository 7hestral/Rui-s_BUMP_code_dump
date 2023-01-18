import json
import os
import ast
import csv
import io
from io import StringIO, BytesIO, TextIOWrapper
import gzip
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import ast
from datetime import timedelta
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
import sys
import time
import torch
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf

def constraint_test(value_list, method="max", threshold=0.05, leq=True):
    if method == "max":
        max_value = max(value_list)
        # print(p_value_list)
        if leq:
            return max_value <= threshold
        else:
            return max_value >= threshold
    if method == "avg":
        print(np.mean(value_list))
        return np.mean(value_list) <= threshold
        if leq:
            return np.mean(value_list) <= threshold
        else:
            return np.mean(value_list) >= threshold
    if method == "min":
        min_value = min(value_list)
        # print(p_value_list)
        if leq:
            return min_value <= threshold
        else:
            return min_value >= threshold

# def get_p_value_list(df, features, start_window, curr_window_size):
#     p_value_list = []
#     for feature in features:
#         series = df[start_window:start_window+curr_window_size][feature]
#         p_value = adfuller(series, autolag='AIC')[1]
#         p_value_list.append(p_value)
#     return p_value_list
    
# def window_select_adf(df, features):
#     minimal_window_size = 7
#     curr_window_size = minimal_window_size
#     total_len = len(df)
#     start_window = 0
#     p_value_list = []
#     break_points = []
#     while start_window + curr_window_size < total_len:
#         p_value_list = get_p_value_list(df, features, start_window, curr_window_size)
#         constraint_satisfied = p_value_constraint_test(p_value_list, "avg")
#         if not constraint_satisfied:
#             break_points.append(start_window)
#             start_window = start_window + curr_window_size
#             curr_window_size = minimal_window_size
#         else:
#             curr_window_size += 1
#     return break_points

# def get_acf_list(df, features, start_window, curr_window_size):
#     acf_value_list = []
#     for feature in features:
#         series = df[start_window:start_window+curr_window_size][feature]
#         acf_values = torch.abs(torch.Tensor(acf(series, nlags=len(series) - 1)))
#         acf_value = np.min(acf_values)
#         acf_value_list.append(acf_value)
#     return acf_value_list
def window_select(df, features, window_selector, minimal_window_size=7):
    curr_window_size = minimal_window_size
    total_len = len(df)
    start_window = 0
    p_value_list = []
    break_points = []
    while start_window + curr_window_size < total_len:
        constraint_satisfied = window_selector(start_window, curr_window_size)
        if not constraint_satisfied:
            break_points.append(start_window)
            start_window = start_window + curr_window_size
            curr_window_size = minimal_window_size
        else:
            curr_window_size += 1
    return break_points


class SingleWindowSelector():
    def __init__(self, df, features):
        self.df = df
        self.features = features
    def __call__(self, start_window, curr_window_size):
        raise NotImplementedError

class ADFSingleWindowSelector(SingleWindowSelector):
    def __init__(self, df, features, p_value_threshold=0.05, method='max'):
        super().__init__(df, features)
        self.p_value_threshold = p_value_threshold
        self.method = method
    def __call__(self, start_window, curr_window_size):
        p_value_list = []
        for feature in self.features:
            series = self.df[start_window:start_window+curr_window_size][feature]
            p_value = adfuller(series, autolag='AIC')[1]
            p_value_list.append(p_value)
        print(p_value_list)
        return constraint_test(p_value_list, method=self.method, threshold=self.p_value_threshold, leq=True)

class ACFSingleWindowSelector(SingleWindowSelector):
    def __init__(self, df, features, correlation_threshold=0.6, method='min'):
        super().__init__(df, features)
        self.correlation_threshold = correlation_threshold
        self.method = method
    def __call__(self, start_window, curr_window_size):
        acf_value_list = []
        for feature in self.features:
            series = self.df[start_window:][feature]
            acf_values = torch.abs(torch.Tensor(acf(series, nlags=len(series) - 1)))
            acf_value_list.append(acf_values[:curr_window_size])
        acf_value_list = torch.stack(acf_value_list)
        
        if self.method == 'min':
            aggregated_acf = torch.min(acf_value_list, axis=0)[0]
        if self.method == 'avg':
            aggregated_acf = torch.mean(acf_value_list, axis=0)
        print(aggregated_acf)
        return torch.min(aggregated_acf).item() >= self.correlation_threshold