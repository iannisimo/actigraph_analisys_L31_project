# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from operator import le
from utils import load_dataset, tables
import pandas as pd
import numpy as np
import pandas_profiling as pdp
from sklearn.model_selection import train_test_split, cross_val_score
from util_classes import custom_cross_val_score_reg

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor, TweedieRegressor, SGDRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor

(X_train, y_train, g_train, _, _, _, _) = load_dataset(
    tables.AHA, tables.METADATA, length=120, overlap=-60, 
    class_method='macs', attrs='DND', scale='',
)
from util_classes import reg_score
# %%
regs = [
    MLPRegressor(),
    RandomForestRegressor(),
    BaggingRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor(),
    LinearRegression(), 
    PoissonRegressor(), 
    TweedieRegressor(), 
    SGDRegressor(),
    SVR(),
    NuSVR()
]
for reg in regs:
    # scores = cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1, scoring=reg_score, groups=g_train)
    scores = custom_cross_val_score_reg(reg, X_train, y_train, groups=g_train, n_jobs=-1)
    print (f'{np.mean(scores)}\t{np.var(scores)}\t{reg.__class__.__name__}\n\t{scores}\n')

# %%
