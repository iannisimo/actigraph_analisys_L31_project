# %%
import os
from turtle import Vec2D
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, f1_score, r2_score
from util_classes import sg_cross_val_score, reg_score, custom_cross_val_score_reg

from sklearn.neural_network import MLPRegressor
from kfold import StratifiedGroupKFold
from termcolor import cprint


fix_inline()


mdl = MLPRegressor(max_iter=10000, hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=1e-2)

for a in ['D', 'ND', 'DND']:
    (X, y, g, _, _, _, _) = load_dataset(
        tables.AHA, tables.METADATA, length=120, overlap=-1, 
        class_method='macs', attrs=a, split=False
    )
    scores = custom_cross_val_score_reg(mdl, X, y, groups=g, n_splits=5, n_jobs=-1)
    cprint(f'ALL:\t{scores}\nMean(STD):\t{np.mean(scores)}({np.std(scores)})', 'green')
