import numpy as np
import pandas as pd

df = pd.read_csv('./EMG_data/01/1_raw_data_13-12_22.03.16.txt', sep="\t")

print(df)
print(df.columns)

df['class'] = df['class'].astype(int)

print(df.dtypes)