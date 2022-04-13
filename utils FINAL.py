# %%
from os import path
from enum import Enum
from typing import Tuple
from matplotlib.pyplot import cla, table
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
import pandas as pd
import math as m
import pandas_profiling as pdp
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, Normalizer

# 3.0 -> 4.0: StratifiedShuffleSplit <- GroupShuffleSplit
# 4.0 -> 5.0:  GroupShuffleSplit <- StratifiedShuffleSplit
# 5.0 -> 5.1:  Added more features
# 5.1 -> 5.2:  Reduced features to quintiles
# Reverted to 5.1
# 5.1 -> 6.0: Shuffle train-test every call
# 6.0 -> 6.1: Reverted to deciles
# 6.1 -> 7.0: Added ['vec_D', 'vec_ND', 'AI'] to attrs
# 7.0 -> 7.1: Data transformation in range [0,1]
# Reverted to 7.0
# 7.0 -> 7.2: Modified quantiles
# 7.2 -> 7.3: More Features
# 7.3 -> 7.4: Added back window values
# 7.4 -> 7.5: Like 7.3
# 7.5 -> 8.0: Overlap == -1 : auto overlap
# 8.0 -> 8.1: GroupShuffleSplit <- StratifiedGroupKFold
# 9.0_a: added subject features
# 10.0: attrs selector
# 11.0: removed subject features duplicate
# 12.0: MAX zeros allowed
# 13.0: Removed side_hemi
# 14.0: Removed hemi, added age_aha
VERSION = '14.0'

def fix_inline():
    get_ipython().run_line_magic('matplotlib', 'inline')

AHA_FILE = "../data/actigraph_aha.ftr"
WEEK_FILE = "../data/actigraph_week.ftr"
METADATA_FILE = "../data/metadata.ftr"

def str2attrs(attrs: str):
    if attrs == 'DND':
        return ['x_D', 'y_D', 'z_D', 'vec_D', 'x_ND', 'y_ND', 'z_ND', 'vec_ND']
    if attrs == 'D':
        return ['x_D', 'y_D', 'z_D', 'vec_D']
    if attrs == 'ND':
        return ['x_ND', 'y_ND', 'z_ND', 'vec_ND']
    if attrs == 'AI':
        return ['AI']
    if attrs == 'DNDAI':
        return ['x_D', 'y_D', 'z_D', 'vec_D', 'x_ND', 'y_ND', 'z_ND', 'vec_ND', 'AI']

def get_class(subject: int, metadata: pd.DataFrame, method: str):
    if method == 'macs':
        return metadata.MACS[metadata.subject == subject].iat[0]
    elif method.startswith('aha'):
        classes = method.replace('aha', '')
        if len(classes) == 0:
            return metadata.AHA[metadata.subject == subject].iat[0]
        elif classes.isdecimal():
            classes = int(classes) - 1
            aha_i = metadata.AHA[metadata.subject == subject].iat[0]
            if aha_i == 100: return 0
            #   Divide all AHA values != 100 in `classes` semi-equal classes 
            aha_classes = np.array_split(np.sort(metadata.AHA[metadata.AHA != 100].values), classes)
            c = [aha_i in x for x in aha_classes].index(True) + 1
            return c
            # return m.ceil(aha_i / (99 / (classes - 1)))
    elif method.startswith('oldaha'):
        classes = method.replace('oldaha', '')
        if len(classes) == 0:
            return metadata.AHA[metadata.subject == subject].iat[0]
        elif classes.isdecimal():
            classes = int(classes)
            aha_i = metadata.AHA[metadata.subject == subject].iat[0]
            if aha_i == 100: return 0
            return np.int64(np.ceil(aha_i / (99 / (classes - 1))))
        else:
            raise('NaN AHA classes')

def get_classes(method: str):
    if method == 'macs':
        return 4
    elif method.startswith('aha'):
        classes = method.replace('aha', '')
        if len(classes) == 0:
            return 100
        elif classes.isdecimal():
            return int(classes)
    elif method.startswith('oldaha'):
        classes = method.replace('oldaha', '')
        if len(classes) == 0:
            return 100
        elif classes.isdecimal():
            return int(classes)

# %%
metadata = pd.read_feather(METADATA_FILE)
aha = pd.read_feather(AHA_FILE)
week = pd.read_feather(WEEK_FILE)


# %%

class tables(Enum):
    METADATA = 0
    AHA = 1
    WEEK = 2
    
    def __init__(self, value) -> None:
        self.metadata = metadata
        self.tables = {
            0: self.metadata,
            1: aha,
            2: week
        }
        self.tableNames = {
            0: 'metadata',
            1: 'aha',
            2: 'week'
        }
        super().__init__()

    def __str__(self):
        return self.tableNames[self.value]

    def table(self):
        return self.tables[self.value]

# %%

FEATURES_FUNCTIONS = lambda x: np.append(
    np.append(
        np.array([
            np.mean(x),
            np.median(x),
            np.var(x),
            np.std(x),
            # np.max(x),
            # np.min(x),
            # np.max(np.abs(np.diff(x))),
            # np.max(np.diff(x)),
            # np.min(np.abs(np.diff(x))),
            # np.min(np.diff(x)),
            # np.size(x) - np.count_nonzero(x)
    # Decili
        ]), 
    np.quantile(x, np.arange(0, 1.05, .05))),
np.quantile(np.diff(x), np.arange(0, 1.05, .05)))

# 8.2_a
# FEATURES_FUNCTIONS = lambda x: np.append(
#         np.array([
#             np.mean(x),
#             np.median(x),
#             np.var(x),
#             np.std(x),
#             # np.max(x),
#             # np.min(x),
#             # np.max(np.abs(np.diff(x))),
#             # np.max(np.diff(x)),
#             # np.min(np.abs(np.diff(x))),
#             # np.min(np.diff(x)),
#             # np.size(x) - np.count_nonzero(x)
#     # Decili
#         ]), 
#     np.quantile(x, np.arange(0, 1.05, .1)))

# 7.1
# def extract_features(data: pd.DataFrame):
#     features = dict()
#     q_data = quantile_transform(np.array(data), n_quantiles=data.shape[0])
#     for i in range(q_data.shape[1]):
#         features[f'{i}'] = FEATURES_FUNCTIONS(q_data[:, i])
#     return features

# 7.0
def extract_features(data: pd.DataFrame):
    features = dict()
    M = data.max().max()
    m = data.min().min()
    data = (data - m) / (M - m)
    for col in data:
        features[col] = FEATURES_FUNCTIONS(data[col].values)
    return features

def find_window(data: pd.DataFrame, start: int, length: int, overlap: int, attrs_str: str, max_zeros: int):
    # attrs = ['x_D', 'y_D', 'z_D', 'x_ND', 'y_ND', 'z_ND', 'vec_D', 'vec_ND', 'AI']
    # 8.2_a
    # 'x_D', 'y_D', 'z_D', 'vec_D', 
    attrs = str2attrs(attrs_str)
    last_acc_time = data.Time.iat[-1] - pd.Timedelta(length)
    while True:
        if start >= data.shape[0]:
            return (-1, None)
        start_time = data.Time.iat[start]
        if start_time + pd.Timedelta(seconds=length) > last_acc_time:
            return (-1, None)
        if data[attrs].iloc[start].abs().sum() == 0: 
            start += 1
            continue
        c_window = data[(data.Time >= start_time) & (data.Time <= start_time + pd.Timedelta(seconds=length))]
        if c_window[c_window[attrs].abs().sum(1) == 0].shape[0] >= max_zeros * c_window.shape[0]:
            start += 1
            continue
        new_start = data.Time[data.Time > start_time + pd.Timedelta(seconds=overlap)]
        if len(new_start) > 0: new_start = new_start.index[0]
        else: new_start = -2
        return (new_start, c_window[attrs])

def get_subject_features(metadata: pd.DataFrame, subject: int):
    features = []
    MAX_AGE = 27
    MIN_AGE = 5
    attrs = ['gender', 'dom']
    features += (metadata[metadata.subject == subject][attrs].iloc[0] - 1).tolist()
    features += ((metadata[metadata.subject == subject][['age_aha']].iloc[0] - MIN_AGE) / (MAX_AGE - MIN_AGE)).tolist()
    return features


def single_subject(s_data: pd.DataFrame, metadata: pd.DataFrame, length: int, class_method: str, attrs_str: str, max_zeros: int, overlap: int) -> Tuple[int, np.array, np.array]:
    s_dataset = list()
    s_group = list()
    s_start_idx = s_data.index[0]
    subject = s_data.subject.iat[0]
    s_class = get_class(subject, metadata, class_method)
    s_features = get_subject_features(metadata, subject)
    start = 0
    i = 0
    while True:
        i+=1
        new_start, window = find_window(s_data, start, length, overlap, attrs_str, max_zeros)
        if new_start == -1: break
        w_list = list()
        w_list += s_features
        features = extract_features(window)
        for col in window:
            w_list += features[col].tolist()
            # w_list += window[col].values.tolist() # Removed time series data from the dataset v3.0, Added back in 7.4
        s_dataset.append(w_list)
        s_group.append(s_class)
        if new_start == -2: break
        start = new_start - s_start_idx
    return (subject, s_dataset, s_group)

def count_windows_per_subject(s_data: pd.DataFrame, metadata: pd.DataFrame, length: int, class_method: str, attrs_str: str, max_zeros: int):
    subject = s_data.subject.iat[0]
    s_class = get_class(subject, metadata, class_method)
    s_start_idx = s_data.index[0]
    start = 0
    i = 0
    while True:
        new_start, _ = find_window(s_data, start, length, length, attrs_str, max_zeros)
        if new_start == -1: break
        i += 1
        if new_start == -2: break
        start = new_start - s_start_idx
    return (s_class, i)

def par_make_dataset(data: pd.DataFrame, metadata: pd.DataFrame, length: int, overlap: int, class_method: str, attrs: str, max_zeros: int) -> Tuple[np.array, np.array, np.array]:
    argList = []
    if overlap == -1:
        for _, m_row in metadata.iterrows():
            s_data = data[data.subject == m_row.subject]
            a = [s_data, metadata, length, class_method, attrs, max_zeros, {}]
            argList.append(a)
        res = Parallel(n_jobs=-1)(delayed(count_windows_per_subject)(*args, **kwargs) for *args, kwargs in argList)
        class_counts = np.array([0] * get_classes(class_method))
        for r in res:
            class_counts[r[0]] += r[1]
        overlaps = np.round(class_counts / np.max(class_counts) * length)
        print(f'Overlaps: {overlaps}')
        for i,al in enumerate(argList):
            al[6] = {'overlap': overlaps[res[i][0]]}
    else:
        for _, m_row in metadata.iterrows():
            s_data = data[data.subject == m_row.subject]
            a = [s_data, metadata, length, class_method, attrs, max_zeros, {'overlap': overlap}]
            argList.append(a)
    results = Parallel(n_jobs=-1)(delayed(single_subject)(*args, **kwargs) for *args, kwargs in argList)
    X = []
    y = []
    group = []
    for res in results:
        X += res[1]
        y += [get_class(res[0], metadata, class_method)] * len(res[2])
        # y += res[2]
        group += [res[0]] * len(res[2])
    return np.array(X), np.array(y), np.array(group)

# def make_dataset(data: pd.DataFrame, metadata: pd.DataFrame, length: int, overlap: int, class_method: str) -> Tuple[np.array, np.array, np.array]:
#     X = list()
#     y = list()
#     group = list()
#     for _, m_row in metadata.iterrows():
#         subject = m_row.subject
#         s_data = data[data.subject == subject]
#         s_start_idx = s_data.index[0]
#         s_class = get_class(subject, metadata, class_method)
#         start = 0
#         i = 0
#         while True:
#             i+=1
#             new_start, window = find_window(s_data, start, length, overlap)
#             if new_start == -1: break
#             w_list = list()
#             features = extract_features(window)
#             for col in window:
#                 w_list += features[col].tolist()
#                 w_list += window[col].values.tolist()
#             X.append(w_list)
#             y.append(s_class)
#             group.append(subject)
#             if new_start == -2: break
#             start = new_start - s_start_idx
#     return np.array(X), np.array(y), np.array(group)

from kfold import StratifiedGroupKFold
from sklearn.base import BaseEstimator
from termcolor import cprint

def ScaleOZ(V):
    for v in V:
        M = max(v)
        m = min(v)
        v = (v - m) / (M - m)

def load_dataset(
        data: tables, metadata: tables or pd.DataFrame, 
        length = 240, overlap = 60, 
        class_method = 'macs', attrs='DND',
        split=True, reload=False, 
        removeControl=False, multiclass=False, 
        scale='', version = None, test_pcentage=.3, max_zeros=.3
    ) -> Tuple[np.array, np.array, np.array, np.array, BaseEstimator]:
    skip_table = isinstance(metadata, pd.DataFrame)
    if version == None: version = VERSION
    filename = f'./datasets/{str(data)}_{length}_{overlap}_{class_method}_{attrs}_z{max_zeros}_v{version}.npz'
    fname = filename.replace('./datasets/', '').replace('.npz', '')
    cprint(f'Loading {fname}', 'blue')
    if path.isfile(filename) and not reload and not skip_table:
        cprint('Found cached', 'green')
        X, y, g = dict(np.load(filename)).values()
    else:
        cprint('Processing...', 'red')
        mdata = metadata if skip_table else metadata.table()
        (X, y, g) = par_make_dataset(data.table(), mdata, length, overlap, class_method, attrs, max_zeros)
        if version == VERSION:
            np.savez(filename, X, y, g)
    cprint('Done', 'green')
    scaler = None
    if scale == 's':
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    if scale == 'n':
        scaler = Normalizer().fit(X)
        X = scaler.transform(X)
    if scale == '10':
        X = ScaleOZ(X)
    if split == False:
        if removeControl:
            idx_nctrl = np.where(y != 0)[0]
            X = X[idx_nctrl]
            y = y[idx_nctrl]
            g = g[idx_nctrl]
        return X, y, g, np.array([]), np.array([]), np.array([]), scaler
    gss = StratifiedGroupKFold(n_splits=round(1/test_pcentage), shuffle=False)
    idx_train, idx_test = [a for a in gss.split(X,y,g)].pop()
    X_train, X_test = [X[x] for x in [idx_train, idx_test]]
    y_train, y_test = [y[x] for x in [idx_train, idx_test]]
    g_train, g_test = [g[x] for x in [idx_train, idx_test]]
    if removeControl:
        idx_nctrl_train = np.where(y_train != 0)[0]
        idx_nctrl_test = np.where(y_test != 0)[0]
        X_train = X_train[idx_nctrl_train]
        y_train = y_train[idx_nctrl_train]
        g_train = g_train[idx_nctrl_train]
        X_test = X_test[idx_nctrl_test]
        y_test = y_test[idx_nctrl_test]
        g_test = g_test[idx_nctrl_test]
    if multiclass and class_method != 'aha':
        y_train_n = np.zeros((y_train.shape[0], np.unique(y_train).size))
        for i, j in enumerate(np.unique(y_train)):
            y_train_n[np.where(y_train == j)[0],i] = 1
        y_test_n = np.zeros((y_test.shape[0], np.unique(y_test).size))
        for i, j in enumerate(np.unique(y_test)):
            y_test_n[np.where(y_test == j)[0],i] = 1
        y_train = y_train_n.astype(np.int)
        y_test = y_test_n.astype(np.int)
    return X_train, y_train, g_train, X_test, y_test, g_test, scaler



# %%

# 5.2
# def load_dataset(data: tables, metadata: tables, length = 240, overlap = 60, class_method = 'macs', reload=False, removeControl=False, multiclass=False, scaled=False) -> Tuple[np.array, np.array, np.array, np.array]:
#     filename = f'./datasets/{str(data)}_{length}_{overlap}_{class_method}_v{VERSION}.npz'
#     if path.isfile(filename) and not reload:
#         X_train, y_train, g_train, X_test, y_test, g_test = dict(np.load(filename)).values()
#     else:
#         (X, y, g) = par_make_dataset(data.table(), metadata.table(), length, overlap, class_method)
#         gss = GroupShuffleSplit(n_splits=1, test_size=.30)
#         idx_train, idx_test = [a for a in gss.split(X,y,g)].pop()
#         X_train, X_test = [X[x] for x in [idx_train, idx_test]]
#         y_train, y_test = [y[x] for x in [idx_train, idx_test]]
#         g_train, g_test = [g[x] for x in [idx_train, idx_test]]
#         # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, stratify=y)
#         np.savez(filename, X_train, y_train, g_train, X_test, y_test, g_test)
#     if removeControl:
#         idx_nctrl_train = np.where(y_train != 0)[0]
#         idx_nctrl_test = np.where(y_test != 0)[0]
#         X_train = X_train[idx_nctrl_train]
#         y_train = y_train[idx_nctrl_train]
#         g_train = g_train[idx_nctrl_train]
#         X_test = X_test[idx_nctrl_test]
#         y_test = y_test[idx_nctrl_test]
#         g_test = g_test[idx_nctrl_test]
#     if scaled:
#         scaler = StandardScaler().fit(X_train)
#         X_train = scaler.transform(X_train)
#         X_test = scaler.transform(X_test)
#     if multiclass and class_method != 'aha':
#         y_train_n = np.zeros((y_train.shape[0], np.unique(y_train).size))
#         for i, j in enumerate(np.unique(y_train)):
#             y_train_n[np.where(y_train == j)[0],i] = 1
#         y_test_n = np.zeros((y_test.shape[0], np.unique(y_test).size))
#         for i, j in enumerate(np.unique(y_test)):
#             y_test_n[np.where(y_test == j)[0],i] = 1
#         y_train = y_train_n
#         y_test = y_test_n
#     return X_train, y_train, g_train, X_test, y_test, g_test


# def load_dataset(data: tables, metadata: tables, length = 240, overlap = 60, class_method = 'macs', reload=False, validation=False) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
#     filename = f'./datasets/{str(data)}_{length}_{overlap}_{class_method}_v{VERSION}.npz'
#     if path.isfile(filename) and not reload:
#         X_train, X_trainval, X_val, X_test, y_train, y_trainval, y_val, y_test = dict(np.load(filename)).values()
#     else:
#         (X, y) = make_dataset(data.table(), metadata.table(), length, overlap, class_method)
#         X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=.15, stratify=y)
#         X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=.15, stratify=y_trainval)
#         np.savez(filename, X_train, X_trainval, X_val, X_test, y_train, y_trainval, y_val, y_test)
#     if validation:
#         return X_train, y_train, X_val, y_val, X_test, y_test
#     else:
#         return X_trainval, y_trainval, np.array([]), np.array([]), X_test, y_test