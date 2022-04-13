from tkinter import E
import numpy as np
import pandas as pd
from termcolor import cprint
from utils import str2attrs, extract_features, get_subject_features

class Subject:
    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame, subject_id: int, attrs: str, attrs_sbj: str):
        self.data = data[data.subject == subject_id]
        self.metadata = metadata[metadata.subject == subject_id]
        self.subject_id = subject_id
        self.attrs = str2attrs(attrs)
        self.attrs_sbj = attrs_sbj
        self.sj_features = get_subject_features(self.metadata, self.subject_id, self.attrs_sbj)
    
    def get_windows(self, length: int, max_zeros = 0.3, step=1):
        self.__idxs__ = []
        self.ds = []
        self.mov = []
        self.fts = []
        self.timestamps = []
        for i in range(0, self.data.shape[0] - length, step):
            c_table = self.data.iloc[i:i+length]
            c_attrs = c_table[self.attrs]
            s_vec = c_table.vec_D.sum()
            self.mov.append(s_vec)
            self.fts.append(self.data.iat[i, 0])
            s_attrs = c_attrs.sum(1)
            if s_attrs.iat[0] == 0:             
                continue
            zeros = (s_attrs == 0).sum(0)
            if zeros > length * max_zeros:
                # cprint(f'Too much zeros: {zeros} / {length}', 'red')
                continue
            self.__idxs__.append(i)
            c_features = extract_features(c_attrs)
            c_ds = np.concatenate([self.sj_features, np.concatenate(list(c_features.values()))])
            # c_ds = np.concatenate(list(c_features.values()))
            self.ds.append(c_ds)
            self.timestamps.append(self.data.iat[i, 0])
        return self.ds
