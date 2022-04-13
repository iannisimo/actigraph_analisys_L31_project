# %%
import os
from pprint import pprint

from matplotlib import pyplot as plt
os.chdir('/home/simone/Tesi/python/scratch/')
from operator import le
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import accuracy_score, plot_confusion_matrix, r2_score, classification_report
from pprint import pprint
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
# fix_inline()
# F SCORE

if __name__ == '__main__':

    (X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
    tables.AHA, tables.METADATA, length=180, overlap=120, 
    class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, 
    attrs_sbj='std', random_state=42
)
    # %%
    automl = AutoSklearnClassifier(
        n_jobs=-1,
        time_left_for_this_task=60,
        per_run_time_limit=60,
        memory_limit=None,
        max_models_on_disc=None,
        # Bellow two flags are provided to speed up calculations
        # Not recommended for a real implementation
        # initial_configurations_via_metalearning=0,
        # smac_scenario_args={'runcount_limit': 1},
    )
    automl.fit(X_train, y_train, dataset_name='AHA_180_120_macs_TS')

    ############################################################################
    # View the models found by auto-sklearn
    # =====================================

    ############################################################################
    # Print the final ensemble constructed by auto-sklearn
    # ====================================================

    pprint(automl.show_models(), indent=4)

    ############################################################################
    # Print statistics about the auto-sklearn run
    # ===========================================

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())

    ############################################################################
    # Get the Score of the final ensemble
    # ===================================

    from util_classes import reg_scorer
    from sklearn.metrics import ConfusionMatrixDisplay

    # %%

    y_pred = automl.predict(X_test)
    y_pred_p = automl.predict_proba(X_test)

    # y_pred_f = automl.predict(X_test)
    # # print("Accuracy score", accuracy_score(y_test, predictions))
    # print("R2 score", r2_score(y_test, y_pred_f))
    # print("RegScorer score", reg_scorer(y_test, y_pred_f))
    # y_pred = y_pred_f.round().clip(0,1)
    # fig,ax = plt.subplots(1,1)
    # sns.scatterplot(y_test, predictions)
    print(classification_report(y_test, y_pred))
    # plot_confusion_matrix(automl, X_test, y_test, ax=ax)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    