# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(tables.AHA, tables.METADATA, length=300, overlap=120, class_method='macs')

# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

mdl = RandomForestClassifier(n_jobs=-1)

space = [
    Integer(2,300, name='n_components'),
    Categorical([True, False], name='whiten')
]

@use_named_args(space)

def objective(**params):
    pca = PCA()
    pca.set_params(**params)
    X_train_pca = pca.fit_transform(X_train)
    return 1.0-np.mean(cross_val_score(mdl, X_train_pca, y_train, cv=5, n_jobs=-1, scoring='accuracy', groups=g_train))

res_gp = gp_minimize(objective, space, n_calls=50, random_state=42)
print("Best score=%.4f" % res_gp.fun)

fix_inline()
plot_convergence(res_gp)
# %% Optimal

# res_gp.x = [23, True]
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

pca = PCA(n_components=23, whiten=True)
X_train_pca = pca.fit_transform(X_train)
opt_cls = RandomForestClassifier()
scores = cross_val_score(opt_cls, X_train_pca, y_train, cv=5, n_jobs=-1, groups=g_train)
print(f'Scores: {scores}')
fitted = opt_cls.fit(X_train, y_train)
predicted = fitted.predict(X_test)

fig, ax = plt.subplots(1,1)
plot_confusion_matrix(fitted, X_test, y_test, ax=ax)
# sns.scatterplot(x=predicted, y=y_test, ax=ax)
 # %%
