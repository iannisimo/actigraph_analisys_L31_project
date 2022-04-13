# %%
from argparse import MetavarTypeHelpFormatter
import pandas as pd
import feather as ft
from typing import Callable

AHA_FILE = "../data/actigraph_aha.csv"
WEEK_FILE = "../data/actigraph_week.csv"
METADATA_FILE = "../data/metadata.csv"
ftrname: Callable[[str], str] = lambda x: x.replace('.csv', '.ftr') 

metadata = pd.read_csv(METADATA_FILE)
aha = pd.read_csv(AHA_FILE)
aha['Time'] = pd.to_datetime(aha['Time'])
week = pd.read_csv(WEEK_FILE)
week['Time'] = pd.to_datetime(week['Time'])

metadata.to_feather(ftrname(METADATA_FILE))
aha.to_feather(ftrname(AHA_FILE))
week.to_feather(ftrname(WEEK_FILE))
# %%
