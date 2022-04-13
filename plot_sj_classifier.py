# %%
import numpy as np
import pandas as pd
from utils import tables, load_dataset, fix_inline
from extract_sj import Subject
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from termcolor import cprint
from util_classes import probs_to_regs, trim_g_pred
from matplotlib import dates

fix_inline()

ATTRS_SBJ = 'std'


def filterNregress(y_pred_p):
    # filtered = []
    regr = []
    for pred_p in y_pred_p:
        argsort = pred_p.argsort()
        if abs(argsort[-1] - argsort[-2]) == 1:
            a = pred_p[argsort[-1]]
            b = pred_p[argsort[-2]]
            s = a + b
            r = argsort[-1] * (a / s) + argsort[-2] * (b / s)
            regr.append(r)
        else:
            if len(regr) > 0:
                regr.append(regr[-1])
            else:
                regr.append(0)
    # return argsort[-1] * a[argsort[-1]] + argsort[-2] * a[argsort[-2]]
    # return np.array(filtered)
    return np.array(regr)


if __name__ == '__main__':
    (X_train, y_train, g_train, _, _, _, scaler) = load_dataset(
        tables.WEEK, tables.METADATA, length=14400, overlap=-5760, random_state=42,
        class_method='m5', attrs='DND', scale='', split=True, max_zeros=.3, attrs_sbj=ATTRS_SBJ, test_pcentage=.25
    )

# %%
    # mdl = MLPRegressor(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=0.005101621000226295)
    # mdl = RandomForestClassifier(bootstrap = False, max_depth = 40, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 2, n_estimators = 710, n_jobs=-1)
    mdl = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=20, n_estimators=25)
    mdl.fit(X_train, y_train)
    
# %% 
    g_test = np.setdiff1d(np.arange(1, 61), g_train)
    cprint(g_test, 'red')
    # g_test = [59]


# %%
    from util_classes import fix_pgf
    # fix_pgf()
    p_test = []
    p_pred_f = []
    # for s in g_test:
    for s in [29]:
    # for s in  range(1,25):
        sj = Subject(tables.WEEK.table(), tables.METADATA.table(), s, "DND", ATTRS_SBJ)
        # if tables.METADATA.table().iat[sj.subject_id - 1, 1] == 0: continue
        cprint(f'Current subject:\t{s}', 'green', attrs=['bold'])
        a = sj.get_windows(61, max_zeros=.3, step=5)
        # (X_test, _, _, _, _, _, _) = load_dataset(
        #     tables.WEEK, sj.metadata, length=12000, overlap=239, 
        #     class_method='macs', attrs='ND', scale='',
        #     split=False
        # )
        # X_test_s = scaler.transform(X_test)
        # y_pred_1 = mdl.predict(X_test)
        # y_pred_p_t = trim_g_pred(y_pred_p, min_confidence=.5)
        # y_pred = mdl.predict(sj.ds)
        # y_pred = probs_to_regs(y_pred_p)
        # y_pred = np.sum(np.array([0,1,2,3])*y_pred_p_t,1)
        y_pred_p = mdl.predict_proba(sj.ds)
        # y_pred = np.sum(np.array([0,1,2,3])*y_pred_p,1)
        y_pred = filterNregress(y_pred_p)
        p_test.append(tables.METADATA.table().iat[sj.subject_id - 1, 1])
        p_pred_f.append(np.median(y_pred))
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 4)
        ax.set_ylim(-.3,3.5)
        # ax.plot(sj.fts, sj.mov / np.max(sj.mov) * 3)
        ax.plot(sj.timestamps, y_pred, label='Predizione modello')
        ax.plot(sj.timestamps, [p_pred_f[-1]] * y_pred.shape[0], label='Mediana predizioni')
        ax.plot(sj.timestamps, [p_test[-1]] * y_pred.shape[0], label='Classe AHA')
        ax.set_ylabel('Classe')
        ax.grid(True, axis='x', linestyle=':')
        ax.legend(loc='upper right')
        ax.tick_params('x', labelrotation=70)
        

        # ax.xaxis.set_minor_locator(dates.HourLocator(interval=8))   # every 4 hours
        # ax.xaxis.set_minor_formatter(dates.DateFormatter('%H'))  # hours and minutes
        ax.xaxis.set_major_locator(dates.HourLocator(interval=3))    # every day
        ax.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m %H:%M'))
        # ax.plot( y_pred)
        # ax.plot([p_test[-1]] * y_pred.shape[0])
        # ax.plot([p_pred_f[-1]] * y_pred.shape[0])
        # plt.show(block=True)
        fig.savefig('plots/regr_approx.png', bbox_inches='tight')
        # plt.savefig('plots/regr_approx.pgf')
    p_pred = np.round(p_pred_f).clip(0,3)
    ConfusionMatrixDisplay.from_predictions(p_test, p_pred)
# %%
