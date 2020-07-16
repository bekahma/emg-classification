import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_extraction import mean_absolute_value
from feature_extraction import variance
from feature_extraction import standard_error
from feature_extraction import root_mean_square
from feature_extraction import slope_sign_change
from feature_extraction import waveform_length
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Extract signal files
"""
from zipfile import ZipFile

file_name1 = "dataframe_no_signal_compressed.zip"
file_name2 = 'labelled_signal_data_epochs_compressed.zip'

with ZipFile(file_name1, 'r') as zip:
    zip.printdir()

    print('Extracting files from dataframe_no_signal_compressed.zip...')
    zip.extractall()
    print('Done')

with ZipFile(file_name2, 'r') as zip:
    zip.printdir()

    print('Extracting files from labelled_signal_data_epochs_compressed.zip...')
    zip.extractall()
    print('Done')
"""

# Load signal files
labelled_signals = np.loadtxt('./labelled_signal_data_epochs.txt')
df = pd.read_csv('./dataframe_no_signal.csv')

# Assemble into dataframe
df['signal'] = labelled_signals.tolist()
df['signal'] = df['signal'].apply(np.array)
df['signal'] = df['signal'].apply(lambda x: 1000*x) # Convert from V to mV

# X data will be the unique IDs, they will allow indexing of the signals properly
# y data will be class labels
cls_1_ids = df.loc[df['label']==1, 'id'].unique()
cls_2_ids = df.loc[df['label']==2, 'id'].unique()
cls_3_ids = df.loc[df['label']==3, 'id'].unique()
# ids and labels is order [1, 1, 1, 1, 1, ..., 1, 2, 2, 2, 2, 2, ..., 2]
cls_ids_12 = np.append(cls_1_ids, cls_2_ids)
cls_ids_23 = np.append(cls_2_ids, cls_3_ids)
cls_labels_12 = np.append(np.ones((cls_1_ids.size,)), np.ones((cls_2_ids.size,)) * 2)
cls_labels_23 = np.append(np.ones((cls_2_ids.size,)) * 2, np.ones((cls_3_ids.size,)) * 3)
# Split data
X_train_12, X_test_12, y_train, y_test = train_test_split(cls_ids_12, cls_labels_12, test_size=0.25, random_state=0)
X_train_23, X_test_23, y_train, y_test = train_test_split(cls_ids_23, cls_labels_23, test_size=0.25, random_state=0)

def get_train_signals_2a(cls, channel):
    """
    :param cls: class label (integer 1 or 2)
    :param channel: channel number (integer between 1 and 8)
    :return: N x 256 array of signals of selected class and channel
    """
    cls_signals = df.loc[(df['id'].isin(X_train_12)) & (df['label'] == cls), 'signal'].values.tolist()
    cls_signals = np.array(cls_signals)
    cls_signals = np.reshape(cls_signals, (cls_signals.shape[0] // 8, 8, 256))
    return cls_signals[:, channel-1, :]


def get_train_signals_2b(cls, channel):
    """
    :param cls: class label (integer 2 or 3)
    :param channel: channel number (integer between 1 and 8)
    :return: N x 256 array of signals of selected class and channel
    """
    cls_signals = df.loc[(df['id'].isin(X_train_23)) & (df['label'] == cls), 'signal'].values.tolist()
    cls_signals = np.array(cls_signals)
    cls_signals = np.reshape(cls_signals, (cls_signals.shape[0] // 8, 8, 256))
    return cls_signals[:, channel-1, :]


def test_classifier_2a(classifier_fun):
    """
    Tests accuracy of manual classifier function on test data for class 1 and 2
    :param classifier_fun: function that returns class based on input 8x256 nd-array of a single signal epoch
    :return: Accuracy score
    """
    test_signals = df.loc[(df['id'].isin(X_test_12)), 'signal'].values.tolist()
    test_signals = np.array(test_signals)
    test_signals = np.reshape(test_signals, (test_signals.shape[0] // 8, 8, 256))

    # Get labels from df. Since it's repeated 8 times for each channel, only pull every 8th one
    ys = df.loc[(df['id'].isin(X_test_12)), 'label'].values
    ys = ys[::8]

    classifier_output = [classifier_fun(signal) for signal in test_signals]
    classifier_output = np.array(classifier_output)
    num_correct = np.count_nonzero(classifier_output == ys)
    accuracy = num_correct / ys.size
    print('{} correct out of {} in test dataset for 2a'.format(num_correct, ys.size))
    print('Classifier accuracy on test data: {:.3f}'.format(accuracy))
    return accuracy


def test_classifier_2b(classifier_fun):
    """
    Tests accuracy of manual classifier function on test data for class 2 and 3
    :param classifier_fun: function that returns class based on input 8x256 nd-array of a single signal epoch
    :return: Accuracy score
    """
    test_signals = df.loc[(df['id'].isin(X_test_23)), 'signal'].values.tolist()
    test_signals = np.array(test_signals)
    test_signals = np.reshape(test_signals, (test_signals.shape[0] // 8, 8, 256))

    # Get labels from df. Since it's repeated 8 times for each channel, only pull every 8th one
    ys = df.loc[(df['id'].isin(X_test_23)), 'label'].values
    ys = ys[::8]

    classifier_output = [classifier_fun(signal) for signal in test_signals]
    classifier_output = np.array(classifier_output)
    num_correct = np.count_nonzero(classifier_output == ys)
    accuracy = num_correct / ys.size
    print('{} correct out of {} in test dataset for 2b'.format(num_correct, ys.size))
    print('Classifier accuracy on test data: {:.3f}'.format(accuracy))
    return accuracy

#class 1 and 2
def classifier_se_var_2a(x):
    """
    Classifies EMG segment, x, as either class 1 or 2
    :param x: 8 by 256 nd-array. Dimension 0 are the 8 channels. Dimension 1 is the signal for each channel
    :return: class label
    """
    feature_1 = standard_error(x[0, :])
    feature_2 = variance(x[0, :])
    if (feature_1 < 0.00175) and (feature_2 < 0.005):
        return 1
    else: return 2

#class 2 and 3
def classifier_mav_rms_2b(x):
    feature_1 = mean_absolute_value(x[0, :])
    feature_2 = root_mean_square(x[0, :])
    if (feature_1 < 0.034) and (feature_2 < 0.3):
        return 1
    else: return 2

# get train signals for class 1,2 channel 1
"""
cls1_ch1 = get_train_signals_2a(1, 1)
cls2_ch1 = get_train_signals_2a(2, 1)
cls1_ch1.shape

cls1_ch1_se = np.apply_along_axis(standard_error, 1, cls1_ch1)
cls2_ch1_se = np.apply_along_axis(standard_error, 1, cls2_ch1)
cls1_ch1_var = np.apply_along_axis(variance, 1, cls1_ch1)
cls2_ch1_var = np.apply_along_axis(variance, 1, cls2_ch1)
cls1_ch1_se.shape
cls1_ch1_var.shape
"""

#get train signals for class 2,3 channel 4
cls2_ch1 = get_train_signals_2b(2, 1)
cls3_ch1 = get_train_signals_2b(3, 1)
# cls2_ch2 = get_train_signals_2b(2, 2)
# cls3_ch1 = get_train_signals_2b(3, 1)
cls2_ch1.shape

cls2_ch1_mav = np.apply_along_axis(mean_absolute_value, 1, cls2_ch1)
cls3_ch1_mav = np.apply_along_axis(mean_absolute_value, 1, cls3_ch1)
cls2_ch1_rms= np.apply_along_axis(root_mean_square, 1, cls2_ch1)
cls3_ch1_rms = np.apply_along_axis(root_mean_square, 1, cls3_ch1)
cls2_ch1_mav.shape
cls3_ch1_rms.shape
#plot class 1,2 channel 1
"""
plt.figure(figsize=(12,8))
plt.scatter(cls1_ch1_se, cls1_ch1_var, c='green', label="Resting Features", s=4)
plt.scatter(cls2_ch1_se, cls2_ch1_var, c='red', label="Fist Features", s=4)
ax = plt.gca()
ax.add_patch(mpatches.Rectangle((0, 0), 0.00175, 0.005, fill = False, color = 'purple'))
plt.legend(loc='best')
plt.xlabel('Standard Error (CH1)')
plt.ylabel('Variance (CH1)')
plt.title("Feature plot for Class 1 (Resting) and Class 2 (Fist)")
plt.show()
"""

#plot class 2,3 channel 4
plt.figure(figsize=(12,8))
plt.scatter(cls2_ch1_mav, cls2_ch1_rms, c='green', label="Fist Feature", s=4)
plt.scatter(cls3_ch1_mav, cls3_ch1_rms, c='red', label="Wrist Flexion Feature", s=4)
ax = plt.gca()
# ax.add_patch(mpatches.Rectangle((0, 0), 0.00175, 0.005, fill = False, color = 'purple'))
plt.legend(loc='best')
plt.xlabel('Mean Absolute Value (CH1)')
plt.ylabel('Root Mean Square (CH1)')
plt.title("Feature plot for Class 2 (Fist) and Class 3 (Wrist Flexion)")
plt.show()

#call test classifier functions
# test_classifier_2a(classifier_se_var_2a)
test_classifier_2b(classifier_mav_rms_2b)


