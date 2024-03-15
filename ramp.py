import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

array_1 = np.ones(360)
array_2 = 2 * array_1
array_3 = 3 * array_1
array_4 = 4 * array_1
array_5 = 5 * array_1
array = np.concatenate((array_1, array_2, array_3),axis=0)
array = np.concatenate((array, array_4, array_5),axis=0)
array = np.tile(array, 7)

seg_file = os.path.join('/Users/xinguofang/Documents/Research/TI-aware Clustering', 'binning', 'TI.mat')
segidx = scipy.io.loadmat(seg_file)
segidx = segidx['Segidx']

plt.subplot(2, 1, 1)
plt.plot(array)

plt.subplot(2, 1, 2)
plt.plot(segidx)

plt.show()