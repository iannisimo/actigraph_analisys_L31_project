# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables
import numpy as np

from sklearn.neural_network import MLPClassifier

from utils import fix_inline
fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=120, overlap=60, class_method='macs', 
    removeControl=False, scale='', multiclass=False, split=True
)

# col = 1
# y_train = y_train[:,col]
# y_test = y_test[:,col]

# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from util_classes import sg_cross_val_score

mdl = MLPClassifier(max_iter=1000)

space = [
    Real(1e-9, 1e+3, name='alpha', prior='log-uniform'),
    Categorical(['tanh','relu'],name='activation'),
    Integer(1, 13, name='n_layers')
]
for i in range(1,14):
    space.append(Integer(20,2000, name=f'layer_{i}', prior='log-uniform'))

@use_named_args(space)

def objective(**params):
    n_layers = params.pop('n_layers')
    params['hidden_layer_sizes'] = ()
    for i in range(1, n_layers+1):
        params['hidden_layer_sizes'] += (params.pop(f'layer_{i}'))
    for i in range(n_layers+1, 14): params.pop(f'layer_{i}')
    mdl.set_params(**params)
    return 1.0-np.mean(sg_cross_val_score(
        mdl, X_train, y_train, n_splits=5
        , n_jobs=-1, scoring='f1_weighted', groups=g_train))

res_gp = gp_minimize(objective, space, n_calls=20, verbose=True)
print("Best score=%.4f" % res_gp.fun)
plot_convergence(res_gp)
# %% Optimal
alpha = 1
# alpha = res_gp.x[0]
activation = res_gp.x[1]
n_layers = res_gp.x[2]
h_layers = tuple(res_gp.x[3:3+n_layers])

# res_gp.x = ['tanh', 9, 1784, 1921, 1813, 348, 798, 644, 1213, 1516, 809, 1525, 20, 1517, 82]
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, f1_score

opt_cls = MLPClassifier(max_iter=1000, activation=activation, hidden_layer_sizes=h_layers, alpha=alpha)
print(opt_cls)
opt_cls.fit(X_train, y_train)
predicted = opt_cls.predict(X_test)
f1 = f1_score(y_test, predicted, average='weighted')
print(f'Test score: {f1}')

from utils import fix_inline
fix_inline()
fig, ax = plt.subplots(1,1)
plot_confusion_matrix(opt_cls, X_test, y_test, ax=ax)
# sns.scatterplot(x=predicted, y=y_test, ax=ax)
 # %%
