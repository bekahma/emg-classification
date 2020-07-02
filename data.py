import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./EMG_data/01/1_raw_data_13-12_22.03.16.txt', sep="\t")

print(df)
print(df.columns)

# convert class column to integer type
df['class'] = df['class'].astype(int)

print("\ndata types of each column: ")
print(df.dtypes)

# extract data from dataFrame to separate numpy arrays
time_col = df.loc[:,'time']
class_col = df.loc[:,'class']
channel1_col = df.loc[:,'channel1']

t = time_col.values
cl = class_col.values
ch = channel1_col.values

print("\nnumpy arrays for time, class, channel1: ")
print(t)
print(cl)
print(ch)

# multiply channel1 data by 1000 (convert to mV)
ch_mv = ch * 1000
print("\nchannel1 data in mV: ")
print(ch_mv)

# divide time data by 1000 (convert to sec)
t_sec = t / 1000
print("\ntime data in sec: ")
print(t_sec)

# plot1: ch_mv vs. t_sec
# plt.plot(t_sec, ch_mv)
# plt.ylabel('channel1 (mV)')
# plt.xlabel('time in (s)')
# plt.title('plot 1: EMG signal vs. time')
# plt.show()

# plot 2: ch_mv vs. t_sec (red when class = 2)
# make an array of strings w the same length as signal (black, black, red..etc)
colour_signal = []
# colour_signal array must be the same length as class array
for i in cl:
    # hand clenched = red colour in plot
    # ^ how do we know h
    if i == 2:
        colour_signal.append('r')
    else:
        colour_signal.append('b')
print("\ncolour_signal:")
print(colour_signal)

# plt.plot(t_sec, ch_mv, color='colour_signal')
# plt.show()

# <----------------- yewon's attempt...? ----------------->
# sections of the signal where class = 2
first_signal = ch_mv[cl == 2]
first_time = t[cl == 2]

plt.plot(first_time, first_signal, color='red')

# sections of the signal where class != 2
first_signal = ch_mv[cl != 2]
first_time = t[cl != 2]

plt.plot(first_time, first_signal, color='blue')

# show the plot for the first signal
plt.show()

# length = len(cl)
# plt.plot(t_sec, ch_mv, color='blue')
# for i in range(length):
#    if cl[i] == 2:
#       plt.plot(t_sec[i], ch_mv[i], color='red')

# plt.title('plot 2: EMG signal vs. time')
# plt.ylabel('channel1 (mV)')
# plt.xlabel('time (s)')
# plt.show()


