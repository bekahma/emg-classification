import numpy as np
import statistics

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

#bekah's feature - SCC
def slope_sign_change(x):
    scc = np.diff(np.sign(np.diff(x)))
    return scc[scc!=0].size

#yewon's feature




#lixin's feature
def standard_error(x):
    sd = statistics.stdev(x)
    sqrt_num = np.sqrt(len(x))
    return sd/sqrt_num



#test array
arr1 = [1,2,-3,4,5]
# slope_sign_change(arr1)




