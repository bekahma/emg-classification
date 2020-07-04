import numpy as np

def mean_absolute_value(x):
    sum1 = sum(np.absolute(x))
    return sum1/len(x)

def root_mean_square(x):
    sum1 = sum(np.square(x))
    return np.sqrt(sum1/len(x))

def variance(x):
    arr = []
    mean = sum(x)/len(x)
    for i in x:
        val = i - mean
        arr.append(val)
    return sum(np.square(arr))/len(x)

#test array
arr1 = [1,2,3,4,5]




