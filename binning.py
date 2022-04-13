# %%
from utils import aha, metadata, fix_inline, week
from matplotlib import pyplot as plt
import numpy as np
from termcolor import cprint
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, classification_report
from sklearn.model_selection import GroupKFold


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def euc_dist(vec_a, vec_b):
    return np.linalg.norm(vec_a - vec_b)


fix_inline()

data = week

train = []
test = []
X = []
y = []
g = []
N_BINS = 20
x = np.arange(-95, 96, 10)
fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
for m in np.unique(metadata.MACS):
    train.append([])
    test.append([])
    c_mdata = metadata[metadata.MACS == m] 
    c_data = data[data.subject.isin(c_mdata.subject.to_list())]
    c_data_z = c_data[(c_data.vec_D != 0) & (c_data.vec_ND != 0)]
    subjects = np.random.permutation(np.unique(c_data_z.subject.to_list()))
    n = max(1, round(len(subjects) * .3))
    s_test = subjects[0:n]
    s_train = subjects[n::]
    ax[m//2, m%2].set_title(f'MACS {m}' if m != 0 else 'Control')
    for s in subjects:
        s_data = c_data_z[c_data_z.subject == s]
        bins = [0] * N_BINS
        for b in range(N_BINS):
            step = 200 / N_BINS
            mi = - 100 + (step * b)
            Ma = mi + step
            if Ma == 100: Ma = 101
            bins[b] = np.sum((s_data.AI >= mi) & (s_data.AI < Ma))
        bins = np.array(bins) / np.mean(bins)
        if s in s_train:
            train[-1].append(bins)
        else:
            test[-1].append(bins)
        ax[m//2, m%2].plot(x, bins)
        X.append(bins)
        y.append(m)
        g.append(s)

# %%
y_true = []
y_pred = []

for m in range(len(test)):
    for s in range(len(test[m])):
        s_class = np.array([0.,0.,0.,0.])
        for mm in range(len(train)):
            sum_dist = 0
            for ss in range(len(train[mm])):
                sum_dist += euc_dist(test[m][s], train[mm][ss])
            sum_dist /= len(train[mm])
            s_class[mm] = sum_dist
        p_class = np.array([0,1,2,3])[s_class == min(s_class)][0]
        y_true.append(m)
        y_pred.append(p_class)
fig.savefig('plots/bins_week.pgf')
# plt.show()
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, colorbar=False, display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
plt.savefig('plots/bins_week_cm.pgf')

# %%

from sklearn.metrics import recall_score, precision_score, f1_score

gkf = GroupKFold(n_splits=5)

X =  np.array(X)
y = np.array(y)
cv = gkf.split(X, y, g)
ty_test = np.array([])
ty_pred = np.array([])
# scores = []
precs = [[], [], [], []]
# accs = []
recs = [[], [], [], []]
f1s = [[], [], [], []]
for split in cv:
    X_train, X_test = X[split[0]], X[split[1]]
    y_train, y_test = y[split[0]], y[split[1]]
    y_pred = []
    for i, xt in enumerate(X_test):
        sum_dist = [0,0,0,0]
        for j, xT in enumerate(X_train):
            sum_dist[y_train[j]] += euc_dist(xt, xT)
        sum_dist = np.array(sum_dist) / np.unique(y_train, return_counts=1)[1]
        p_class = np.array([0,1,2,3])[sum_dist == min(sum_dist)][0]
        y_pred.append(p_class)
    y_pred = np.array(y_pred)
    ty_test = np.append(ty_test, y_test)
    ty_pred = np.append(ty_pred, y_pred)
    for m in np.unique(y):
        prec = precision_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
        reca = recall_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
        f1 = f1_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
        precs[m].append(prec)
        recs[m].append(reca)
        f1s[m].append(f1)
    # scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    # precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
    # accs.append(accuracy_score(y_test, y_pred))
    # recs.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    # cprint(classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=['Control', 'MACS 1', 'MACS 2', 'MACS 3'], zero_division=0))
# for m in np.unique(y):
#     ms = 'Control:' if m == 0 else f'MACS {m}:'
#     cprint(f'{ms} & {np.mean(precs[m]):.2f} & {np.mean(recs[m]):.2f} \\\\', 'green')
t = '\t' * 2
cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
for m in np.unique(y):
    ms = 'Control:' if m == 0 else f'MACS {m}:'
    cprint(f'{t}{ms} & {np.mean(precs[m]):.2f} & {np.mean(recs[m]):.2f} & {np.mean(f1s[m]):.2f} \\\\', 'green')
cprint(f'{t}\\hline\n{t}Media & {np.mean(precs):.2f} & {np.mean(recs):.2f} & {np.mean(f1s):.2f} \\\\\n{t}\\hline', 'green')

ConfusionMatrixDisplay.from_predictions(ty_test, ty_pred, colorbar=False, display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
plt.savefig('plots/bins_week_cv_cm.pgf')

# cprint(f'{scores}\n\t{np.mean(scores)}({np.std(scores)})')
# cprint(f'{precs}\n\t{np.mean(precs)}({np.std(precs)})')
# cprint(f'{accs}\n\t{np.mean(accs)}({np.std(accs)})')
# cprint(f'{recs}\n\t{np.mean(recs)}({np.std(recs)})')

# %%
