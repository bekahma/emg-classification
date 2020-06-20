import numpy as np
import pandas as pd

df = pd.read_csv("./EMG_data/01/1_raw_data_13-12_22.03.16.txt")

print(df)
print(df.columns)
print(type(df.columns))

df["columns"] = df["columns"].to_numeric(int)
