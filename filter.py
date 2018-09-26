from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import sys


if __name__ == '__main__':
    std_ratio = 1.

    q = 0.001
    r = 0.001

    rootgrp = Dataset('data/sp500.nc', 'r')
    logreturns = rootgrp.variables['y'][:]
    logreturns = rootgrp.variables['y'][:]

    logreturns = logreturns[:100]
    plt.plot(logreturns)

    kalman = KalmanFilter(1, 1)
    kalman.x = np.zeros((kalman.dim_x, 1))
    kalman.P = np.eye(kalman.dim_x) * 0.001 ** 2

    kalman.Q = q ** 2
    kalman.R = np.eye(kalman.dim_z) * r ** 2
    kalman.F = np.eye(kalman.dim_x)

    filtered = []
    covs = []
    kalman.predict()

    filtered.append(kalman.x[0, 0])
    covs.append(kalman.P[0, 0])
    for r in logreturns:
        # prediction
        kalman.update(np.array([r]), H=np.eye(1))
        kalman.predict()
        filtered.append(kalman.x[0, 0])
        covs.append(kalman.P[0, 0])

    k_stds = np.sqrt(covs)
    plt.plot(filtered, color='red')
    plt.plot(filtered + k_stds, color='red', alpha=0.3)
    plt.plot(filtered - k_stds, color='red', alpha=0.3)

    x = os.system('./filter.sh')
    if x:
        sys.exit(x)

    libbi_filtered_grp = Dataset('filtered.nc', 'r')
    libbi_logreturns = libbi_filtered_grp.variables['x'][:]

    plt.plot(libbi_logreturns.mean(axis=1), 'green')
    std = libbi_logreturns.std(axis=1)
    plt.plot(libbi_logreturns.mean(axis=1) + std, 'green', alpha=0.3)
    plt.plot(libbi_logreturns.mean(axis=1) - std, 'green', alpha=0.3)

    plt.show()

    plt.hist(libbi_logreturns[18, :], bins=30)
    plt.show()

    plt.plot(std)
    plt.plot(k_stds, color='red')
    plt.show()
