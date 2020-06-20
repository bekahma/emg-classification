import numpy as np
import pandas as pd

file1 = pd.read_csv("./EMG_data/01/1_raw_data_13-12_22.03.16.txt")

print(file1.describe())

# file1.close()

# print file1.describe()