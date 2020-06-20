import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# multiply channel1 data by 1000 (convert to mV)
ch_mv = ch * 1000
print(ch_mv)

# divide time data by 1000 (convert to sec)
t_sec = t / 1000
print(t_sec)

# plot 1: ch_mv vs. t_sec
plt.plot(t_sec, ch_mv)
plt.ylabel('channel1 (mV)')
plt.xlabel('time in (s)')
plt.title('plot 1: EMG signal vs. time')
# plt.show()

# plot 2: ch_mv vs. t_sec (red when class = 2)
length = len(cl)
# plt.plot(t_sec, ch_mv, color='blue')
for i in range(length):
    if cl[i] == 2:
        plt.plot(t_sec[i], ch_mv[i], color='red')

plt.title('plot 2: EMG signal vs. time')
plt.ylabel('channel1 (mV)')
plt.xlabel('time (s)')
plt.show()


