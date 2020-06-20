import numpy as np
import pandas as pd

df = pd.read_csv('./EMG_data/01/1_raw_data_13-12_22.03.16.txt', sep="\t")

print(df)
print(df.columns)

df['class'] = df['class'].astype(int)

print(df.dtypes)

time_col = df.loc[:,'time']
class_col = df.loc[:,'class']
channel1_col = df.loc[:,'channel1']

t = time_col.values
cl = class_col.values
ch = channel1_col.values

print(t)
print(cl)
print(ch)