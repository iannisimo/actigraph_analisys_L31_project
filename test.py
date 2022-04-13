# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
from util_classes import sg_cross_val_score
import pandas as pd
import numpy as np
import pandas_profiling as pdp

from sklearn.metrics import check_scoring
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold, GroupKFold
from kfold import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler

fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=240, overlap=180, 
    class_method='macs', 
    removeControl=False, multiclass=False)

# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_train = normalize(X_train)

# # %%
# cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
# for split in cv.split(X_train, y_train, groups=g_train):
#     tr, te  = split
#     print(np.intersect1d(g_train[tr], g_train[te]))
#     print(len(g_train[te]) / len(g_train[tr]))
#     print(np.unique(g_train[tr], return_counts=1))
#     print(np.unique(g_train[te], return_counts=1))
#     print(f'tr_y:\t{y_train[tr]}')
#     print(f'te_y:\t{y_train[te]}')
#     print(np.unique(y_train[tr], return_counts=1))
#     print(np.unique(y_train[te], return_counts=1))


# %%
mdl = KerasClassifier()
# scorer = check_scoring(mdl, scoring='accuracy')

# cv = StratifiedKFold(n_splits=5)
sg_cross_val_score(mdl, X_train, y_train, n_splits=5, n_jobs=-1, scoring='accuracy', groups=g_train)
# cross_validate(
#         estimator=mdl,
#         X=X_train,
#         y=y_train,
#         groups=g_train,
#         scoring={"score": scorer},
#         cv=cv,
#         n_jobs=-1,
#         return_estimator=True,
#         return_train_score=True,
#     )
# %%
