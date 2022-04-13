# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import classification_report, f1_score
from util_classes import sg_cross_val_score
from kfold import StratifiedGroupKFold
gkf = GroupKFold(n_splits=3)

from sklearn.svm import SVC

# import matplotlib
# matplotlib.use('TkAgg')
fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=100, overlap=100, class_method='macs', 
    removeControl=True, scaled=True, multiclass=False
)
# %%
# col = 3
# y_train = y_train[:,col]
# y_test = y_test[:,col]

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

mdl = BayesSearchCV(SVC(max_iter=100000), {
        'C': (1e-6, 1e+1, 'log-uniform'),
        'gamma': (1e-6, 1e+3, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=32,
    cv=gkf,
    verbose=10,
    scoring='f1_weighted',
    n_jobs=-1,

)

mdl.fit(X_train, y_train, groups=g_train)
y_pred = mdl.predict(X_test)

# %%

print("val. score: %s" % mdl.best_score_)
print("test score: %s" % f1_score(y_test, y_pred, average='weighted'))

# %%

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

predicted = mdl.predict(X_test)
print(classification_report(y_test, predicted))

fig, ax = plt.subplots(1,1)
plot_confusion_matrix(mdl, X_test, y_test, ax=ax)
# %%
