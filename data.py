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
channel2_col = df.loc[:,'channel2']
channel3_col = df.loc[:,'channel3']

t = time_col.values
cl = class_col.values
ch = channel1_col.values
ch2 = channel2_col.values
ch3 = channel3_col.values

print("\nnumpy arrays for time, class, channel1: ")
print(t)
print(cl)
print(ch)

# multiply channel data by 1000 (convert to mV)
ch_mv = ch * 1000
print("\nchannel1 data in mV: ")
print(ch_mv)

ch_mv2 = ch2 * 1000
ch_mv3 = ch3 * 1000

# divide time data by 1000 (convert to sec)
t_sec = t / 1000
print("\ntime data in sec: ")
print(t_sec)

# plot1: ch_mv vs. t_sec
plt.plot(t_sec, ch_mv)
plt.ylabel('channel1 (mV)')
plt.xlabel('time in (s)')
plt.title('plot 1: EMG signal vs. time')
plt.show()

# plot 2: ch_mv vs. t_sec (red when class = 2)
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

# plot 3: channel 1 to 3 plot (same colour)
plt.figure()
plt.subplot(311)
plt.ylabel('channel1')
plt.plot(t_sec, ch_mv)

plt.subplot(312)
plt.ylabel('channel2')
plt.plot(t_sec, ch_mv2)

plt.subplot(313)
plt.ylabel('channel3')
plt.plot(t_sec, ch_mv3)
plt.xlabel('time in s')
# plt.show()

# plot 4: channel 1 to 3 plot (red when hand is clenched)
plt.figure()
plt.subplot(311)
plt.ylabel('channel1')

fs = ch_mv[cl == 2]
ft = t[cl == 2]
plt.plot(ft, fs, color='red')

fs = ch_mv[cl != 2]
ft = t[cl != 2]

plt.plot(ft, fs, color='blue')

plt.subplot(312)
plt.ylabel('channel2')

fs2 = ch_mv2[cl == 2]
ft2 = t[cl == 2]
plt.plot(ft2, fs2, color='red')

fs2 = ch_mv2[cl != 2]
ft2 = t[cl != 2]

plt.plot(ft2, fs2, color='blue')

plt.subplot(313)
plt.ylabel('channel3')

fs3 = ch_mv3[cl == 2]
ft3 = t[cl == 2]
plt.plot(ft3, fs3, color='red')

fs3 = ch_mv3[cl != 2]
ft3 = t[cl != 2]

plt.plot(ft3, fs3, color='blue')

plt.xlabel('time in s')
plt.show()

