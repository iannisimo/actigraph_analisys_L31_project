from unittest import makeSuite
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.metrics import check_scoring, f1_score, make_scorer
from sklearn.base import BaseEstimator
from kfold import StratifiedGroupKFold
import numpy as np
from joblib import Parallel, delayed

from kfold import StratifiedGroupKFold

from utils import fix_inline

import matplotlib

def fix_pgf():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    # scores = cross_val_score(mdl, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy', groups=g_train)

def sg_cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, n_splits=5, n_jobs=None, verbose=0, fit_params=None, pre_dispatch="2*n_jobs", error_score=np.nan):
    scorer = check_scoring(estimator, scoring=scoring)
    # cv = GroupKFold(n_splits=n_splits)
    cv = StratifiedGroupKFold(n_splits=n_splits)

    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring={"score": scorer},
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
    )
    # res = cross_validate(estimator=estimator, X=X, y=y, groups=groups, cv=cv, n_jobs=n_jobs, return_estimator=1, return_train_score=1)
    return cv_results['test_score']

def subject_score_proba(classes, y_true, y_proba, *, groups):
    u_groups = np.unique(groups)
    p_pred = []
    p_test = []
    for g in u_groups:
        g_idx = np.where(groups == g)
        y_proba_g = y_proba[g_idx]
        sum_proba = np.median(y_proba_g,0)
        p_class_idx = np.unravel_index(sum_proba.argmax(), sum_proba.shape)[0]
        p_pred += [classes[p_class_idx]]
        p_test += [y_true[g_idx[0][0]]]
    return f1_score(p_test, p_pred, average='macro')

def subject_pred(y_true, y_pred_f, groups):
    u_groups = np.unique(groups)
    p_pred = []
    p_true = []
    for g in u_groups:
        g_idx = np.where(groups == g)
        y_pred_g = y_pred_f[g_idx]
        p_pred_g = np.mean(y_pred_g,0).round().clip(0,3)
        p_pred += [p_pred_g]
        p_true += [y_true[g_idx[0][0]]]
    return (p_true, p_pred)

def subject_pred_conf(y_true, y_pred_f, groups):
    u_groups = np.unique(groups)
    u_classes = np.sort(np.unique(y_true))
    p_pred = []
    p_true = []
    for g in u_groups:
        g_idx = np.where(groups == g)
        y_pred_g = y_pred_f[g_idx]
        scores = [0] * (u_classes.shape[0])
        for p in y_pred_g:
            nearest_int = np.round(p).clip(0,3)
            rem = np.abs(p - nearest_int)
            if rem <= .2:
                score = 1
            else:
                score = 0
                # score = min(1, - np.log10(rem) - .3)
            scores[int(nearest_int)] += score
        scores = np.array(scores)
        p_pred_g = u_classes[scores == scores.max()][0]
        p_pred += [p_pred_g]
        p_true += [y_true[g_idx[0][0]]]
    return (p_true, p_pred)

def subject_pred_prob(y_true, y_proba, groups):
    u_groups = np.unique(groups)
    p_pred = []
    p_true = []
    for g in u_groups:
        g_idx = np.where(groups == g)
        y_pred_g = y_proba[g_idx]
        p_true += [y_true[g_idx[0][0]]]
        sum_pred = np.sum(y_pred_g, 0)
        p_pred_g = np.sort(np.unique(y_true))[sum_pred == max(sum_pred)][0]
        p_pred += [p_pred_g]
    return (p_true, p_pred)

def fit_estimate_proba(estimator, classes, X_train, y_train, X_test, y_test, g_test):
    estimator.fit(X_train, y_train)
    y_proba = estimator.predict_proba(X_test)
    return subject_score_proba(classes, y_test, y_proba, groups=g_test)

def custom_cross_val_score(estimator, X, y=None, *, groups=None, n_splits=5, n_jobs=None, verbose=0, fit_params=None):
    cv = GroupKFold(n_splits=n_splits)
    scores = []
    classes = np.unique(y)
    arg_list = []
    for idx_train, idx_test in cv.split(X, y, groups=groups):
        X_train, X_test = [X[x] for x in [idx_train, idx_test]]
        y_train, y_test = [y[x] for x in [idx_train, idx_test]]
        _, g_test = [groups[x] for x in [idx_train, idx_test]]
        arg_list.append([estimator, classes, X_train, y_train, X_test, y_test, g_test, {}])
    scores = Parallel(n_jobs=-1)(delayed(fit_estimate_proba)(*args, **kwargs) for *args, kwargs in arg_list)
    return scores

def subject_score_reg(classes, y_true, y_pred_f, *, groups):
    u_groups = np.unique(groups)
    p_pred = []
    p_test = []
    for g in u_groups:
        g_idx = np.where(groups == g)
        y_pred_g = y_pred_f[g_idx]
        p_pred_g = np.median(y_pred_g,0).round().clip(0,3)
        p_pred += [p_pred_g]
        p_test += [y_true[g_idx[0][0]]]
    return f1_score(p_test, p_pred, average='macro')

def fit_estimate_reg(estimator, classes, X_train, y_train, X_test, y_test, g_test, sw):
    estimator.fit(X_train, y_train)
    y_pred_f = estimator.predict(X_test)
    return subject_score_reg(classes, y_test, y_pred_f, groups=g_test)

def custom_cross_val_score_reg(estimator, X, y=None, *, groups=None, n_splits=5, n_jobs=None, sample_weight=None):
    cv = GroupKFold(n_splits=n_splits)
    scores = []
    classes = np.unique(y)
    arg_list = []
    for idx_train, idx_test in cv.split(X, y, groups=groups):
        X_train, X_test = [X[x] for x in [idx_train, idx_test]]
        y_train, y_test = [y[x] for x in [idx_train, idx_test]]
        _, g_test = [groups[x] for x in [idx_train, idx_test]]
        arg_list.append([estimator, classes, X_train, y_train, X_test, y_test, g_test, sample_weight, {}])
    scores = Parallel(n_jobs=n_jobs)(delayed(fit_estimate_reg)(*args, **kwargs) for *args, kwargs in arg_list)
    return scores

def reg_scorer(y_true, y_pred):
    e = np.abs(y_true - y_pred)
    score = np.mean(e)
    return score

reg_score = make_scorer(reg_scorer, greater_is_better=False)

from sklearn.metrics import precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from termcolor import cprint
from matplotlib import pyplot as plt
from kfold import StratifiedGroupKFold

def proba_to_idx(proba):
    uy = np.arange(0,len(proba))
    return (uy[proba == np.max(proba)][0])

def trim_g_pred(g_pred, min_confidence):
    while True:
        g_pred_trim = []
        for conf in g_pred:
            if max(conf) > np.sum(conf) * min_confidence:
                g_pred_trim.append(conf)
        if len(g_pred_trim) > 0:
            return g_pred_trim
        min_confidence -= 0.05

def savefig_companion(filename, save=False):
    ws = 'w' if 'windows' in filename else 's'
    plt.savefig(f"plots/latex/{ws}_latest.pgf")
    if save==False:
        return
    plt.savefig(filename)

from skopt.utils import use_named_args
from skopt import gp_minimize

def nested_CV(estimator, X, y, groups, space, n_splits_test, n_splits_val, random_state=None, shuffle=False, min_confidence=0.0, tabs=0):
    fix_inline()
    fix_pgf()
    cvt = StratifiedGroupKFold(n_splits=n_splits_test, random_state=None, shuffle=False)
    tp_test = np.array([])
    tp_pred = np.array([])
    c = np.unique(y).shape[0]
    p_precs = [[] for x in range(c)]
    p_recs = [[] for x in range(c)]
    p_f1s = [[] for x in range(c)]
    ty_test = np.array([])
    ty_pred = np.array([])
    precs = [[] for x in range(c)]
    recs = [[] for x in range(c)]
    f1s = [[] for x in range(c)]
    iteration = 0
    for idx_trainval, idx_test in cvt.split(X, y, groups=groups):
        cprint(f'Current outer split: {iteration}', 'green', attrs=['bold'])
        iteration += 1
        X_trainval, X_test = [X[x] for x in [idx_trainval, idx_test]]
        y_trainval, y_test = [y[x] for x in [idx_trainval, idx_test]]
        g_trainval, g_test = [groups[x] for x in [idx_trainval, idx_test]]
        @use_named_args(space)
        def cross_val(**params):
            print(params)
            if np.intersect1d(g_test, g_trainval):
                cprint(f'{np.intersect1d(g_test, g_trainval)}', 'red')
            estimator.set_params(**params)
            cvv = StratifiedGroupKFold(n_splits=n_splits_val, random_state=random_state, shuffle=shuffle)
            scores = []
            for idx_train, idx_val in cvv.split(X_trainval, y_trainval, groups=g_trainval):
                X_train, X_val = [X_trainval[x] for x in [idx_train, idx_val]]
                y_train, y_val = [y_trainval[x] for x in [idx_train, idx_val]]
                g_train, g_val = [g_trainval[x] for x in [idx_train, idx_val]]
                estimator.fit(X_train, y_train)
                
                y_pred = estimator.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='weighted')
                scores.append(f1)

                # y_pred_p = estimator.predict_proba(X_val)
                # p_test = []
                # p_pred = []
                # for g in np.unique(g_val):
                #     g_pred = y_pred_p[g_val == g]
                #     g_pred_trim = trim_g_pred(g_pred, min_confidence)
                #     s_pred = np.median(g_pred_trim, 0)
                #     p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
                #     p_test.append(y_val[g_val == g][0])
                # f1 = f1_score(p_test, p_pred, average='macro')
                # scores.append(f1)

            return 1.0 - np.mean(scores)
        res_gp = gp_minimize(cross_val, space, n_calls=10, verbose=True, n_jobs=-1)
        best_params = {}
        for i, x in enumerate(res_gp.x):
            best_params[space[i].name] = x
        print(best_params)
        estimator.set_params(**best_params)
        print(estimator)
        estimator.fit(X_trainval, y_trainval)
        y_pred_p = estimator.predict_proba(X_test)
        y_pred = estimator.predict(X_test)
        ty_test = np.append(ty_test, y_test)
        ty_pred = np.append(ty_pred, y_pred)
        for m in np.unique(y):
            prec = precision_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            reca = recall_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            f1 =   f1_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            precs[m].append(prec)
            recs[m].append(reca)
            f1s[m].append(f1)
        p_test = []
        p_pred = []
        for g in np.unique(g_test):
            g_pred = y_pred_p[g_test == g]
            g_pred_trim = trim_g_pred(g_pred, min_confidence)
            s_pred = np.median(g_pred_trim, 0)
            p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
            p_test.append(y_test[g_test == g][0])
        tp_test = np.append(tp_test, p_test)
        tp_pred = np.append(tp_pred, p_pred)
        for m in np.unique(y):
            prec = precision_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            reca = recall_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            f1 =   f1_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            p_precs[m].append(prec)
            p_recs[m].append(reca)
            p_f1s[m].append(f1)
    
    t = '\t' * tabs
    
    ConfusionMatrixDisplay.from_predictions(ty_test, ty_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'], colorbar=False)
    
    cprint('Finestre', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(precs[m]):.2f} & {np.mean(recs[m]):.2f} & {np.mean(f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(precs):.2f} & {np.mean(recs):.2f} & {np.mean(f1s):.2f} \\\\\n{t}\\hline', 'green')

    ConfusionMatrixDisplay.from_predictions(tp_test, tp_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'], colorbar=False)

    cprint('Soggetti', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(p_precs[m]):.2f} & {np.mean(p_recs[m]):.2f} & {np.mean(p_f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(p_precs):.2f} & {np.mean(p_recs):.2f} & {np.mean(p_f1s):.2f} \\\\\n{t}\\hline', 'green')

    return p_precs, p_recs, p_f1s, tp_test, tp_pred


def CV(estimator, X, y, groups, space, n_splits_val, random_state=None, shuffle=False, min_confidence=0.0, tabs=0):
    fix_pgf()
    fix_inline()
    @use_named_args(space)
    def cross_val(**params):
        print(params)
        estimator.set_params(**params)
        cvv = StratifiedGroupKFold(n_splits=n_splits_val, random_state=random_state, shuffle=shuffle)
        scores = []
        for idx_train, idx_val in cvv.split(X, y, groups=groups):
            X_train, X_val = [X[x] for x in [idx_train, idx_val]]
            y_train, y_val = [y[x] for x in [idx_train, idx_val]]
            g_train, g_val = [groups[x] for x in [idx_train, idx_val]]
            estimator.fit(X_train, y_train)
            
            y_pred = estimator.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            scores.append(f1)
            # y_pred_p = estimator.predict_proba(X_val)
            # p_test = []
            # p_pred = []
            # for g in np.unique(g_val):
            #     g_pred = y_pred_p[g_val == g]
            #     g_pred_trim = trim_g_pred(g_pred, min_confidence)
            #     s_pred = np.median(g_pred_trim, 0)
            #     p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
            #     p_test.append(y_val[g_val == g][0])
            # f1 = f1_score(p_test, p_pred, average='macro')
            # scores.append(f1)
        return 1.0 - np.mean(scores)
    res_gp = gp_minimize(cross_val, space, n_calls=10, verbose=True, n_jobs=-1)
    best_params = {}
    for i, x in enumerate(res_gp.x):
        best_params[space[i].name] = x
    print(best_params)
    estimator.set_params(**best_params)
    print(estimator)
    return estimator

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from utils import load_dataset, tables
from sklearn.metrics import classification_report

def sCV(space, n_splits_val, random_state=None, shuffle=False, min_confidence=0.0, tabs=0):
    fix_pgf()
    fix_inline()
    @use_named_args(space)
    def cross_val(**params):
        # try:
        l = params.pop('length')
        # l *= 240
        op = params.pop('overlap')
        o = round(l * (op))
        cprint(f'L:{l}, O:{o}\t\top:{op}', 'green')
        cprint(params, 'red')
        if o == 0:
            return 1.0
        (X, y, groups, _, _, _, filename) = load_dataset(
            tables.AHA, tables.METADATA, length=l, overlap=o, 
            class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
        )
        estimator = RandomForestClassifier(n_jobs=-1, random_state=42)
        estimator.set_params(**params)
        cvv = StratifiedGroupKFold(n_splits=n_splits_val, random_state=random_state, shuffle=shuffle)
        scores = []
        for idx_train, idx_val in cvv.split(X, y, groups=groups):
            X_train, X_val = [X[x] for x in [idx_train, idx_val]]
            y_train, y_val = [y[x] for x in [idx_train, idx_val]]
            g_train, g_val = [groups[x] for x in [idx_train, idx_val]]
            estimator.fit(X_train, y_train)
            y_pred_p = estimator.predict_proba(X_val)
            p_val = []
            p_pred = []
            for g in np.unique(g_val):
                g_pred = y_pred_p[g_val == g]
                g_pred_trim = trim_g_pred(g_pred, min_confidence)
                s_pred = np.median(g_pred_trim, 0)
                p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
                p_val.append(y_val[g_val == g][0])
            f1 = f1_score(p_val, p_pred, average='weighted')
            # f1 = f1_score(p_val, p_pred, average='micro')
            # cprint(classification_report(p_val, p_pred, zero_division=0), 'yellow')
            scores.append(f1)
            # y_pred_p = estimator.predict_proba(X_val)
            # p_test = []
            # p_pred = []
            # for g in np.unique(g_val):
            #     g_pred = y_pred_p[g_val == g]
            #     g_pred_trim = trim_g_pred(g_pred, min_confidence)
            #     s_pred = np.median(g_pred_trim, 0)
            #     p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
            #     p_test.append(y_val[g_val == g][0])
            # f1 = f1_score(p_test, p_pred, average='macro')
            # scores.append(f1)
        return 1.0 - np.mean(scores)
        # except:
            # return 1.0
    res_gp = gp_minimize(cross_val, space, n_calls=200, verbose=True, n_jobs=-1)
    # best_params = {}
    # for i, x in enumerate(res_gp.x):
    #     best_params[space[i].name] = x
    # print(best_params)
    return res_gp.x

from sklearn.tree import DecisionTreeClassifier


def sCVlcv(space, n_splits_val):
    fix_pgf()
    fix_inline()
    @use_named_args(space)
    def cross_val(**params):
        # try:
        l = params.pop('length')
        # l *= 240
        op = params.pop('overlap')
        o = round(l * (op))
        cprint(f'L:{l}, O:{o}\t\top:{op}', 'green')
        cprint(params, 'red')
        if o == 0:
            return 1.0
        (X, y, groups, _, _, _, filename) = load_dataset(
            tables.AHA, tables.METADATA, length=l, overlap=o, 
            class_method='aha2', attrs='ND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
        )
        estimator = RandomForestClassifier(n_jobs=-1, random_state=42)
        # estimator = DecisionTreeClassifier(random_state=42)
        estimator.set_params(**params)
        return 1.0 - latex_cross_val(estimator, X, y, groups, n_splits_val, no_show=True)
        # except:
            # return 1.0
    res_gp = gp_minimize(cross_val, space, n_calls=200, verbose=True, n_jobs=-1)
    # best_params = {}
    # for i, x in enumerate(res_gp.x):
    #     best_params[space[i].name] = x
    # print(best_params)
    return res_gp.x

from sklearn.ensemble import RandomForestRegressor

def sCVlcvreg(space, n_splits_val):
    fix_pgf()
    fix_inline()
    @use_named_args(space)
    def cross_val(**params):
        # try:
        l = params.pop('length')
        # l *= 240
        op = params.pop('overlap')
        o = round(l * (op))
        cprint(f'L:{l}, O:{o}\t\top:{op}', 'green')
        cprint(params, 'red')
        if o == 0:
            return 1.0
        (X, y, groups, _, _, _, filename) = load_dataset(
            tables.AHA, tables.METADATA, length=l, overlap=o, 
            class_method='aha2', attrs='ND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
        )
        estimator = RandomForestRegressor(n_jobs=-1, random_state=42)
        estimator.set_params(**params)
        return 1.0 - latex_cross_val_r(estimator, X, y, groups, n_splits_val, no_show=True)
        # except:
            # return 1.0
    res_gp = gp_minimize(cross_val, space, n_calls=200, verbose=True, n_jobs=-1)
    # best_params = {}
    # for i, x in enumerate(res_gp.x):
    #     best_params[space[i].name] = x
    # print(best_params)
    return res_gp.x



def latex_cross_val(estimator, X, y, groups=None, n_splits=5, n_jobs=-1, tabs=0, min_confidence = 0.0, basename=None, save=False, no_show=False):
    fix_pgf()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False, random_state=42)
    ty_test = np.array([])
    ty_pred = np.array([])
    tp_test = np.array([])
    tp_pred = np.array([])
    c = np.unique(y).shape[0]
    precs = [[] for i in range(c)]
    recs = [[] for i in range(c)]
    p_precs = [[] for i in range(c)]
    p_recs = [[] for i in range(c)]
    f1s = [[] for i in range(c)]
    p_f1s = [[] for i in range(c)]
    for idx_train, idx_test in cv.split(X, y, groups=groups):
        X_train, X_test = [X[x] for x in [idx_train, idx_test]]
        y_train, y_test = [y[x] for x in [idx_train, idx_test]]
        g_train, g_test = [groups[x] for x in [idx_train, idx_test]]
        estimator.fit(X_train, y_train)
        y_pred_p = estimator.predict_proba(X_test)
        # y_pred = np.apply_along_axis(proba_to_idx, 0, np.transpose(y_pred_p))
        y_pred = estimator.predict(X_test)
        ty_test = np.append(ty_test, y_test)
        ty_pred = np.append(ty_pred, y_pred)
        for m in np.unique(y):
            prec = precision_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            reca = recall_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            f1 =   f1_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            precs[m].append(prec)
            recs[m].append(reca)
            f1s[m].append(f1)
        p_test = []
        p_pred = []
        for g in np.unique(g_test):
            g_pred = y_pred_p[g_test == g]
            # s_pred = g_pred.sum(0)
            # g_pred_trim = []
            # for conf in g_pred:
            #     if max(conf) > np.sum(conf) * min_confidence:
            #         g_pred_trim.append(conf)
            # if(len(g_pred) != len(g_pred_trim)): cprint(len(g_pred) - len(g_pred_trim), 'red', attrs=['bold'])
            g_pred_trim = trim_g_pred(g_pred, min_confidence)
            s_pred = np.median(g_pred_trim, 0)
            p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
            p_test.append(y_test[g_test == g][0])
        tp_test = np.append(tp_test, p_test)
        tp_pred = np.append(tp_pred, p_pred)
        for m in np.unique(y):
            prec = precision_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            reca = recall_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            f1 =   f1_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            p_precs[m].append(prec)
            p_recs[m].append(reca)
            p_f1s[m].append(f1)
        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
        # plt.show(block=True)
        # ConfusionMatrixDisplay.from_predictions(p_test, p_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
        # plt.show(block=True)

        # cprint(f'Precisions:\t{precs}\nRecalls:   \t{recs}', 'red')
    if no_show:
        # return np.mean(f1s)
        return np.mean(p_f1s)

    d_labels = [f'Class {x}' if x > 0 else 'Control' for x in np.unique(y)]
    labels = [x for x in np.unique(y)]

    ConfusionMatrixDisplay.from_predictions(ty_test, ty_pred, labels=labels, display_labels=d_labels, colorbar=False)
    # plt.show(block=True)
    # plt.savefig(f'plots/windows_{basename}.pgf')
    savefig_companion(f'plots/windows_{basename}.pgf', save)
    ConfusionMatrixDisplay.from_predictions(tp_test, tp_pred, labels=labels, display_labels=d_labels, colorbar=False)
    # plt.show(block=True)
    # plt.savefig(f'plots/subjects_{basename}.pgf')
    savefig_companion(f'plots/subjects_{basename}.pgf', save)
    t = '\t' * tabs
    cprint('Finestre', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(precs[m]):.2f} & {np.mean(recs[m]):.2f} & {np.mean(f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(precs):.2f} & {np.mean(recs):.2f} & {np.mean(f1s):.2f} \\\\\n{t}\\hline', 'green')
    cprint('Soggetti', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(p_precs[m]):.2f} & {np.mean(p_recs[m]):.2f} & {np.mean(p_f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(p_precs):.2f} & {np.mean(p_recs):.2f} & {np.mean(p_f1s):.2f} \\\\\n{t}\\hline', 'green')
    return np.mean(p_f1s)

def latex_cross_val_r(estimator, X, y, groups=None, n_splits=5, n_jobs=-1, tabs=0, min_confidence = 0.0, basename=None, save=False, no_show=False):
    fix_pgf()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False, random_state=42)
    ty_test = np.array([])
    ty_pred = np.array([])
    tp_test = np.array([])
    tp_pred = np.array([])
    c = np.unique(y).shape[0]
    precs = [[] for i in range(c)]
    recs = [[] for i in range(c)]
    p_precs = [[] for i in range(c)]
    p_recs = [[] for i in range(c)]
    f1s = [[] for i in range(c)]
    p_f1s = [[] for i in range(c)]
    for idx_train, idx_test in cv.split(X, y, groups=groups):
        X_train, X_test = [X[x] for x in [idx_train, idx_test]]
        y_train, y_test = [y[x] for x in [idx_train, idx_test]]
        g_train, g_test = [groups[x] for x in [idx_train, idx_test]]
        estimator.fit(X_train, y_train)
        # y_pred = np.apply_along_axis(proba_to_idx, 0, np.transpose(y_pred_p))
        y_pred_f = estimator.predict(X_test)
        y_pred = np.round(y_pred_f).clip(min(y), max(y))
        ty_test = np.append(ty_test, y_test)
        ty_pred = np.append(ty_pred, y_pred)
        for m in np.unique(y):
            prec = precision_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            reca = recall_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            f1 =   f1_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            precs[m].append(prec)
            recs[m].append(reca)
            f1s[m].append(f1)
        p_test = []
        p_pred = []
        for g in np.unique(g_test):
            g_pred = y_pred_f[g_test == g]
            # s_pred = g_pred.sum(0)
            # g_pred_trim = []
            # for conf in g_pred:
            #     if max(conf) > np.sum(conf) * min_confidence:
            #         g_pred_trim.append(conf)
            # if(len(g_pred) != len(g_pred_trim)): cprint(len(g_pred) - len(g_pred_trim), 'red', attrs=['bold'])
            s_pred = np.median(g_pred, 0)
            p_pred.append(np.round(s_pred).clip(min(y), max(y)))
            p_test.append(y_test[g_test == g][0])
        tp_test = np.append(tp_test, p_test)
        tp_pred = np.append(tp_pred, p_pred)
        for m in np.unique(y):
            prec = precision_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            reca = recall_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            f1 =   f1_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            p_precs[m].append(prec)
            p_recs[m].append(reca)
            p_f1s[m].append(f1)
        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
        # plt.show(block=True)
        # ConfusionMatrixDisplay.from_predictions(p_test, p_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'])
        # plt.show(block=True)

        # cprint(f'Precisions:\t{precs}\nRecalls:   \t{recs}', 'red')
    if no_show:
        # return np.mean(f1s)
        return np.mean(p_f1s)

    d_labels = [f'Class {x}' if x > 0 else 'Control' for x in np.unique(y)]
    labels = [x for x in np.unique(y)]

    ConfusionMatrixDisplay.from_predictions(ty_test, ty_pred, labels=labels, display_labels=d_labels, colorbar=False)
    # plt.show(block=True)
    # plt.savefig(f'plots/windows_{basename}.pgf')
    savefig_companion(f'plots/windows_{basename}.pgf', save)
    ConfusionMatrixDisplay.from_predictions(tp_test, tp_pred, labels=labels, display_labels=d_labels, colorbar=False)
    # plt.show(block=True)
    # plt.savefig(f'plots/subjects_{basename}.pgf')
    savefig_companion(f'plots/subjects_{basename}.pgf', save)
    t = '\t' * tabs
    cprint('Finestre', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(precs[m]):.2f} & {np.mean(recs[m]):.2f} & {np.mean(f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(precs):.2f} & {np.mean(recs):.2f} & {np.mean(f1s):.2f} \\\\\n{t}\\hline', 'green')
    cprint('Soggetti', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(p_precs[m]):.2f} & {np.mean(p_recs[m]):.2f} & {np.mean(p_f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(p_precs):.2f} & {np.mean(p_recs):.2f} & {np.mean(p_f1s):.2f} \\\\\n{t}\\hline', 'green')
    return np.mean(p_f1s)

def test_class(estimator, X, y, groups, X_t, y_t, groups_t, basename, tabs=0):
    estimator.fit(X, y)
    y_pred = estimator.predict(X_t)
    y_pred_p = estimator.predict_proba(X_t)
    p_test = []
    p_pred = []
    for g in np.unique(groups_t):
        g_pred = y_pred_p[groups_t == g]
        g_pred_trim = trim_g_pred(g_pred, 0.0)
        s_pred = np.median(g_pred_trim, 0)
        p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
        p_test.append(y_t[groups_t == g][0])
    precs = precision_score(y_t, y_pred, average=None, zero_division=0)
    recs = recall_score(y_t, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_t, y_pred, average=None, zero_division=0)
    p_precs = precision_score(p_test, p_pred, average=None, zero_division=0)
    p_recs = recall_score(p_test, p_pred, average=None, zero_division=0)
    p_f1s = f1_score(p_test, p_pred, average=None, zero_division=0)
    t = '\t' * tabs

    d_labels = [f'Class {x}' if x > 0 else 'Control' for x in np.unique(y)]
    
    ConfusionMatrixDisplay.from_predictions(y_t, y_pred, labels=np.unique(y), display_labels=d_labels, colorbar=False)
    savefig_companion(f'plots/test_windows_{basename}.pgf', True)
    ConfusionMatrixDisplay.from_predictions(p_test, p_pred, labels=np.unique(y), display_labels=d_labels, colorbar=False)
    savefig_companion(f'plots/test_subjects_{basename}.pgf', True)

    cprint('Finestre', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {precs[m]:.2f} & {recs[m]:.2f} & {f1s[m]:.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(precs):.2f} & {np.mean(recs):.2f} & {np.mean(f1s):.2f} \\\\\n{t}\\hline', 'green')
    cprint('Soggetti', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {p_precs[m]:.2f} & {p_recs[m]:.2f} & {p_f1s[m]:.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(p_precs):.2f} & {np.mean(p_recs):.2f} & {np.mean(p_f1s):.2f} \\\\\n{t}\\hline', 'green')


def latex_cross_val_reg(estimator, X, y, groups=None, n_splits=5, n_jobs=-1, tabs=0, basename=None):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False, random_state=42)
    ty_test = np.array([])
    ty_pred = np.array([])
    tp_test = np.array([])
    tp_pred = np.array([])
    precs = [[], [], [], []]
    recs = [[], [], [], []]
    p_precs = [[], [], [], []]
    p_recs = [[], [], [], []]
    f1s = [[], [], [], []]
    p_f1s = [[], [], [], []]
    for idx_train, idx_test in cv.split(X, y, groups=groups):
        X_train, X_test = [X[x] for x in [idx_train, idx_test]]
        y_train, y_test = [y[x] for x in [idx_train, idx_test]]
        _, g_test = [groups[x] for x in [idx_train, idx_test]]
        estimator.fit(X_train, y_train)
        # !!!!!!!!!!!!!!!
        y_pred_f = estimator.predict(X_test)
        y_pred = np.round(y_pred_f).clip(min(y), max(y))
        ty_test = np.append(ty_test, y_test)
        ty_pred = np.append(ty_pred, y_pred)
        # !!!!!!!!!!!!!!!
        # y_pred = estimator.predict(X_test)
        # y_pred_p = estimator.predict_proba(X_test)
        # y_pred_f = probs_to_regs(y_pred_p)
        # !!!!!!!!!!!!!!!
        for m in np.unique(y):
            prec = precision_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            reca = recall_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            f1 =   f1_score((y_test == m) * 1, (y_pred == m) * 1, zero_division=0)
            precs[m].append(prec)
            recs[m].append(reca)
            f1s[m].append(f1)
        p_test = []
        p_pred = []
        for g in np.unique(g_test):
            g_pred = y_pred_f[g_test == g]
            s_pred = np.median(g_pred)
            # s_pred = g_pred.sum(0)
            # g_pred_trim = []
            # for conf in g_pred:
            #     if max(conf) > np.sum(conf) * .0:
            #         g_pred_trim.append(conf)
            # if(len(g_pred) != len(g_pred_trim)): cprint(len(g_pred) - len(g_pred_trim), 'red', attrs=['bold'])
            # s_pred = np.median(g_pred_trim, 0)
            # p_pred.append(np.unique(y)[s_pred == max(s_pred)][0])
            p_pred.append(np.round(s_pred).clip(min(y), max(y)))
            p_test.append(y_test[g_test == g][0])
        tp_test = np.append(tp_test, p_test)
        tp_pred = np.append(tp_pred, p_pred)
        for m in np.unique(y):
            prec = precision_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            reca = recall_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            f1 =   f1_score((p_test == m) * 1, (p_pred == m) * 1, zero_division=0)
            p_precs[m].append(prec)
            p_recs[m].append(reca)
            p_f1s[m].append(f1)
        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'], colorbar=False)
        # plt.show(block=True)
        # ConfusionMatrixDisplay.from_predictions(p_test, p_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'], colorbar=False)
        # plt.show(block=True)    
    
    ConfusionMatrixDisplay.from_predictions(ty_test, ty_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'], colorbar=False)
    savefig_companion(f'plots/windows_{basename}.pgf', True)
    ConfusionMatrixDisplay.from_predictions(tp_test, tp_pred, labels=[0,1,2,3], display_labels=['Control', 'MACS 1', 'MACS 2', 'MACS 3'], colorbar=False)
    savefig_companion(f'plots/subjects_{basename}.pgf', True)

        # cprint(f'Precisions:\t{precs}\nRecalls:   \t{recs}', 'red')
    t = '\t' * tabs
    cprint('Finestre', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(precs[m]):.2f} & {np.mean(recs[m]):.2f} & {np.mean(f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(precs):.2f} & {np.mean(recs):.2f} & {np.mean(f1s):.2f} \\\\\n{t}\\hline', 'green')
    cprint('Soggetti', 'red', attrs=['bold'])
    cprint(f'{t}\\hline\n{t}Classe & Precisione & Recall & F1 \\\\\n{t}\\hline', 'green')
    for m in np.unique(y):
        ms = 'Control:' if m == 0 else f'MACS {m}:'
        cprint(f'{t}{ms} & {np.mean(p_precs[m]):.2f} & {np.mean(p_recs[m]):.2f} & {np.mean(p_f1s[m]):.2f} \\\\', 'green')
    cprint(f'{t}\\hline\n{t}Media & {np.mean(p_precs):.2f} & {np.mean(p_recs):.2f} & {np.mean(p_f1s):.2f} \\\\\n{t}\\hline', 'green')





def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

def get_bins(x, n=20, r=(0,1)):
    m, M = r
    s = (M - m) / n
    bins = np.array([0] * n)
    for i in range(n):
        o = 1
        if i >= n-1: 
            o = 2
        bins[i] = ((x >= (m + (s * i))) & (x < (m + (s * (i + o))))).sum()
    return scale(bins)

def proba_to_reg(proba: np.array):
    proba = scale(proba)
    max_b, max_a = np.sort(proba)[-2::]
    class_a = np.arange(0,len(proba))[proba == max_a]
    if len(class_a) > 1:
        class_b = class_a[1]
    else:
        class_b = np.arange(0,len(proba))[proba == max_b][0]
    class_a = class_a[0]
    asy = ((max_a - max_b) / (max_a + max_b)) * .5
    return np.mean([class_a, class_b]) + asy * (class_a - class_b)

def probs_to_regs(probs):
    return np.apply_along_axis(proba_to_reg, 1, probs)