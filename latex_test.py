# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from utils import load_dataset, tables, fix_inline
from util_classes import latex_cross_val, latex_cross_val_reg, nested_CV, CV, sCV, sCVlcvreg, latex_cross_val_r
from termcolor import cprint


fix_inline()


(X, y, groups, X_t, y_t, groups_t, filename) = load_dataset(
    tables.AHA, tables.METADATA, length=60, overlap=60, 
    class_method='aha2', attrs='ND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
)


# (X, y, groups, X_t, y_t, groups_t, filename) = load_dataset(
#     tables.WEEK, tables.METADATA, length=28800, overlap=-23040, 
#     class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
# )

# (X, y, groups, X_t, y_t, groups_t, filename) = load_dataset( 
#     tables.WEEK, tables.METADATA, length=43200, overlap=8640, 
#     class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
# )

# (X, y, groups, X_test, y_test, g_test, filename) = load_dataset(
#     tables.WEEK, tables.METADATA, length=24000, overlap=18000, 
#     class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
# )


# (X, y, g, _, _, _, filename) = load_dataset( 
#     tables.WEEK, tables.METADATA, length=28800, overlap=14400, 
#     class_method='macs', attrs='DND', scale='', split=False, max_zeros=.3, attrs_sbj='std'
# )

# (X, y, g, _, _, _, filename) = load_dataset( 
#     tables.WEEK, tables.METADATA, length=86400, overlap=-43200, 
#     class_method='macs', attrs='DND', scale='', split=False, max_zeros=.3, attrs_sbj='std'
# )

# mdl = SVC(probability=True)
# mdl = SVR(kernel='poly', C=1, degree = 4, gamma=1e+1)
# mdl = LinearSVR(epsilon=1e-4, max_iter=10000)

# mdl = RandomForestClassifier(bootstrap = True, max_depth = None, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 20, n_estimators= 550, n_jobs=-1)
# mdl = RandomForestClassifier(bootstrap = True, max_depth = 90, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 674, n_jobs=-1)


# mdl = RandomForestClassifier(bootstrap = True, max_depth = 90, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 10, n_estimators = 1583, n_jobs=-1)


# mdl = RandomForestClassifier(n_jobs=-1)
# mdl = RandomForestRegressor(n_jobs=-1)

# mdl = BaggingClassifier(n_jobs=-1)
# mdl = BaggingRegressor(n_jobs=-1)

# mdl = MLPClassifier(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=3.5148054574095396e-08, max_iter=1000)

# mdl = MLPClassifier(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=0.014540059313439942, max_iter=1000)

# mdl = MLPClassifier(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=0.06373563718966359, max_iter=1000)
# mdl = MLPClassifier(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=5.127813307715843e-6, max_iter=1000)

# mdl = MLPClassifier(hidden_layer_sizes=(100, 20, 20), max_iter=1000)

# mdl = MLPClassifier(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=1e-8, max_iter=1000)

# mdl = RandomForestClassifier(bootstrap= True, max_depth= None, max_features= 'auto', min_samples_leaf= 2, min_samples_split= 2, n_estimators= 422)
# mdl = RandomForestClassifier(bootstrap= True, max_depth= 100, max_features= 'auto', min_samples_leaf= 2, min_samples_split= 2, n_estimators= 508)
# mdl = RandomForestClassifier(bootstrap= False, max_depth= 10, max_features= 'auto', min_samples_leaf= 2, min_samples_split= 40, n_estimators= 4000)
# mdl = RandomForestClassifier(bootstrap= True, max_depth= 40, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 2000)
# mdl = RandomForestClassifier(max_depth= 5, min_samples_leaf= 20, n_estimators= 200, random_state=42)
# mdl = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)
# mdl = RandomForestClassifier(max_depth= 20, min_samples_leaf= 20, n_estimators= 50, random_state=42)
# mdl = RandomForestClassifier(
#     max_depth= 10, 
#     min_samples_leaf= 20, 
#     n_estimators= 25, 
#     random_state=42
# )
mdl = RandomForestClassifier(
    max_depth= 5, 
    min_samples_leaf= 5, 
    n_estimators= 100, 
    random_state=42,
    n_jobs=-1
)

basename = str(mdl).split('(')[0].lower() + '_' + filename.replace('.npz', '').split('/')[-1]
print(basename)

cprint(f'Validation {basename}')
# mdl = MLPRegressor(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=0.005101621000226295)
# mdl = MLPRegressor(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=1e-5)

cprint(mdl, 'blue', attrs=['bold'])

# latex_cross_val(mdl, X, y, g, n_splits=5, tabs=2, min_confidence=0.0, basename=basename, save=True)
# from skopt.space import Real, Integer, Categorical

# space = [
#     # Categorical(['mlp', 'rfr'], name='model'),
#     # Categorical(['rfr'], name='model'),
#     Categorical([60, 120, 180, 240, 300, 360], name='length'),
#     Categorical([1/5, -1/5, 1/4, -1/4, 1/3, -1/3, 1/2, -1/2, 1, -1], name='overlap'),

#     # Categorical([(100, ), (100, 100, ), (100, 20, ), (100, 20, 20)], name='hidden_layer_sizes'),
#     # Categorical(['relu', 'tanh'], name='activation'),
#     # Real(1e-10, 1e+1, prior='log-uniform', name='alpha'),

#     Categorical([True, False], name='bootstrap'),
#     Categorical([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, None], name='max_depth'),
#     Categorical(['auto', 'sqrt'], name='max_features'),
#     Categorical([1,2,4,8,16,32], name='min_samples_leaf'),
#     Categorical([2,5,10,20,30,40,50,70,100], name='min_samples_split'),
#     Integer(100, 4000, name='n_estimators')
# ]

# nested_CV(mdl, X, y, g, space, 5, 5)

# opt = CV(mdl, X, y, groups, space, 5)
# x = sCV(space, 5, tabs=2)

# opt = RandomForestClassifier(n_estimators=1989, bootstrap=True, max_depth=None, max_features='auto', min_samples_leaf=4, min_samples_split=20)

# latex_cross_val_r(mdl, X, y, groups, n_splits=5, tabs=2, basename=basename, save=True)
latex_cross_val(mdl, X, y, groups, n_splits=5, tabs=2, basename=basename, save=True)

# %%
from util_classes import test_class
cprint(f'Test {basename}')

test_class(mdl, X, y, groups, X_t, y_t, groups_t, basename=basename, tabs=2)

# mdl.fit(X, y)
# y_pred = mdl.predict(X_t)
# y_pred_p = mdl.predict_proba(X_t)
# p_test = []
# p_pred = []
# for g in np.unique(groups_t):
#     g_pred = y_pred_p[groups_t == g]
#     # g_pred_trim = trim_g_pred(g_pred, 0.0)
#     g_pred_trim = g_pred
#     s_pred = np.median(g_pred_trim, 0)
#     p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
#     p_test.append(y_t[groups_t == g][0])

# from sklearn.metrics import classification_report

# cprint(classification_report(y_t, y_pred, zero_division=0), 'red')
# cprint(classification_report(p_test, p_pred, zero_division=0), 'green')

# latex_cross_val_reg(mdl, X, y, g, n_splits=5, tabs=3, basename=basename)
# %%

# from util_classes import trim_g_pred
# from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# # opt = RandomForestClassifier(bootstrap=False, max_depth=30, min_samples_leaf=2,
#                     #    min_samples_split=5, n_estimators=891, n_jobs=-1)

# cprint(opt, 'blue')

# opt.fit(X, y)

# y_pred = opt.predict(X_t)
# y_pred_p = opt.predict_proba(X_t)

# p_test = []
# p_pred = []
# for g in np.unique(groups_t):
#     g_pred = y_pred_p[groups_t == g]
#     g_pred_trim = trim_g_pred(g_pred, 0.0)
#     s_pred = np.median(g_pred_trim, 0)
#     p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
#     p_test.append(y_t[groups_t == g][0])
# ConfusionMatrixDisplay.from_predictions(y_t, y_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
# ConfusionMatrixDisplay.from_predictions(p_test, p_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
# cprint(classification_report(y_t, y_pred, zero_division=0), 'green', attrs=['bold'])
# cprint(classification_report(p_test, p_pred, zero_division=0), 'green', attrs=['bold'])
# %%

from utils import load_dataset, tables, fix_inline
from util_classes import latex_cross_val, latex_cross_val_reg, nested_CV, CV, sCV, sCVlcv, sCVlcvreg
from termcolor import cprint


fix_inline()

from skopt.space import Real, Integer, Categorical

space = [
    # Categorical(['mlp', 'rfr'], name='model'),
    # Categorical(['rfr'], name='model'),
    # Categorical([120], name='length'),
    Categorical([60, 120, 180, 240, 300, 360], name='length'),
    # Categorical([1/3, -1/3, 1/2, -1/2, 2/3, -2/3, 1, -1], name='overlap'),
    # Categorical([1/2, -1/2, 2/3, -2/3, 1, -1], name='overlap'),
    # Categorical([-2/5, -3/5, -4/5, -1], name='overlap'),
    Categorical([-2/5, -3/5, -4/5, -1, 2/5, 3/5, 4/5, 1], name='overlap'),
    # Categorical([-1], name='overlap'),

    # Categorical([(100, ), (100, 100, ), (100, 20, ), (100, 20, 20)], name='hidden_layer_sizes'),
    # Categorical(['relu', 'tanh'], name='activation'),
    # Real(1e-10, 1e+1, prior='log-uniform', name='alpha'),

    # Categorical([True, False], name='bootstrap'),
    # Categorical([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None], name='max_depth'),
    # Categorical(['auto', 'sqrt'], name='max_features'),
    # Categorical([1,2,4,8], name='min_samples_leaf'),
    # Categorical([2,5,10,20], name='min_samples_split'),
    # Integer(100, 2000, name='n_estimators')
    Categorical([2,3,5,10,20], name='max_depth'),
    Categorical([5,10,20,50,100,200], name='min_samples_leaf'),

    Categorical([10,25,30,50,100,200], name='n_estimators')
]


sCVlcv(space, 5)
# %%
