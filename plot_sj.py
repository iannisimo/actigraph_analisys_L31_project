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

fix_inline()


if __name__ == '__main__':
    (X_train, y_train, g_train, _, _, _, scaler) = load_dataset(
        tables.WEEK, tables.METADATA, length=12000, overlap=6000, 
        class_method='macs', attrs='DND', scale='',
        test_pcentage=.25
    )

# %%
    mdl = MLPRegressor(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=0.005101621000226295)
    # mdl = RandomForestClassifier()
    mdl.fit(X_train, y_train)
    
# %% 
    g_test = np.setdiff1d(np.arange(1, 61), g_train)
    cprint(g_test, 'red')
    # g_test = [59]

    p_test = []
    p_pred_f = []

# %%
    for s in g_test:
    # for s in  range(1,61):
        sj = Subject(tables.WEEK.table(), tables.METADATA.table(), s, "DND")
        if tables.METADATA.table().iat[sj.subject_id - 1, 1] == 0: continue
        cprint(f'Current subject:\t{s}', 'green', attrs=['bold'])
        a = sj.get_windows(51, max_zeros=1, step=5)
        # (X_test, _, _, _, _, _, _) = load_dataset(
        #     tables.WEEK, sj.metadata, length=12000, overlap=239, 
        #     class_method='macs', attrs='ND', scale='',
        #     split=False
        # )
        # X_test_s = scaler.transform(X_test)
        # y_pred = mdl.predict(sj.ds)
        # y_pred_1 = mdl.predict(X_test)
        y_pred = mdl.predict(sj.ds)
        p_test.append(tables.METADATA.table().iat[sj.subject_id - 1, 1])
        p_pred_f.append(np.median(y_pred))
        fig, ax = plt.subplots()
        # ax.plot(y_pred_1)
        # ax.plot(y_pred)
        ax.set_ylim(-1,4)
        ax.plot(sj.timestamps, y_pred)
        ax.plot(sj.timestamps, [p_test[-1]] * y_pred.shape[0])
        ax.plot(sj.timestamps, [p_pred_f[-1]] * y_pred.shape[0])
        plt.show(block=True)
    p_pred = np.round(p_pred_f).clip(0,3)
    ConfusionMatrixDisplay.from_predictions(p_test, p_pred)
# %%
