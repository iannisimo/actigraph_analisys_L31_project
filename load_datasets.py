# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from operator import le
from utils import load_dataset, tables, fix_inline
import pandas as pd
import numpy as np
import pandas_profiling as pdp
from sklearn.model_selection import train_test_split, cross_val_score

settings = [
    [tables.WEEK, tables.METADATA, 7200, 3600, 'aha2'],
    [tables.WEEK, tables.METADATA, 7200, 3600, 'aha3'],
    [tables.WEEK, tables.METADATA, 7200, 3600, 'aha4'],
    [tables.WEEK, tables.METADATA, 7200, 3600, 'aha'],
    [tables.WEEK, tables.METADATA, 7200, 3600, 'macs'],
    [tables.WEEK, tables.METADATA, 7200, 7200, 'aha2'],
    [tables.WEEK, tables.METADATA, 7200, 7200, 'aha3'],
    [tables.WEEK, tables.METADATA, 7200, 7200, 'aha4'],
    [tables.WEEK, tables.METADATA, 7200, 7200, 'aha'],
    [tables.WEEK, tables.METADATA, 7200, 7200, 'macs'],
    [tables.AHA, tables.METADATA, 120, 60, 'aha'],
    [tables.AHA, tables.METADATA, 120, 60, 'aha2'],
    [tables.AHA, tables.METADATA, 120, 60, 'aha3'],
    [tables.AHA, tables.METADATA, 120, 60, 'aha4'],
    [tables.AHA, tables.METADATA, 120, 60, 'macs'],
    [tables.AHA, tables.METADATA, 120, 120, 'aha'],
    [tables.AHA, tables.METADATA, 120, 120, 'aha2'],
    [tables.AHA, tables.METADATA, 120, 120, 'aha3'],
    [tables.AHA, tables.METADATA, 120, 120, 'aha4'],
    [tables.AHA, tables.METADATA, 120, 120, 'macs'],
]

for setting in settings:
    print(f'Next:\n\t{setting}')
    load_dataset(*setting)
    print('Done')
# %%
