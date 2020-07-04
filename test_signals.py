# Code to generate test signals for Task List 4
import numpy as np
import pandas as pd

# Make sure this points to the right directory. 
# I have all the folders (01, 02, etc.) in a directory called 'data'.
file = 'data/01/1_raw_data_13-12_22.03.16.txt'

# Load the data into a dataframe
data = np.loadtxt(file, skiprows=1)
df = pd.DataFrame(data, columns=['time', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'class'])
df['class'] = df['class'].astype(int)

# Get test signals in units of mV
# Use test_1 and test_2 to report values for your feature functions to test if they are working correctly
test_1 = df.loc[df['class'] == 1, 'ch1'].values * 1000
test_2 = df.loc[df['class'] == 2, 'ch1'].values * 1000

# Generate several test signals to make example feature plot
window_len = 300
n_iter = 10 
signal_windows = {}

for signal, label in zip([test_1, test_2], ['resting', 'fist']):
    lst_windows = []
    for i in range(n_iter):
        lst_windows.append(np.array(signal[i*window_len:(i+1)*window_len]))
    signal_windows[label] = np.array(lst_windows)


# signal_windows is a dictionary with the keys 'resting' and 'fist'
# Each entry in the dictionary is a nd-array 
# of shape (10, 300) containing 10 windows of length 300 each