import copy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import pandas as pd
import os
import random
import matplotlib
from sklearn.linear_model import LinearRegression
from pylab import mpl
#matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

def get_file(path, envlog, datalen):
    files = os.listdir(path)
    files_all = {}
    for file in files:
        kk = file.split(envlog)
        name = file.split('_')
        if kk[-1] == datalen+'.log':
            files_all[name[0]] = path+"\\"+file
    return files_all


def get_data(name):
    data = pd.read_csv(name, header=None)[1:]
    data_file = {}
    for i, dat in enumerate(data.values):
        kkk = dat[0].split(' ')
        danow = []
        for j in kkk:
            if len(j) > 0:
                danow.append(j)
        data_file[i] = danow
    return data_file


def moving_average(x, window=3):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # 补齐长度
    front_pad = x[:window//2]
    back_pad = x[-(window - window//2 - 1):] if (window - window//2 - 1) > 0 else []
    return np.concatenate([front_pad, ma, back_pad])[:len(x)]


import numpy as np
from sklearn.linear_model import LinearRegression
import copy

import numpy as np

def data_deal(data, limit):
    """
    Extends a short time series to the specified length using trend-based extrapolation.
    Parameters:
        data (list or array): Input time series data.
        limit (int): Target length to extend to.
    Returns:
        extended (np.ndarray): Extended sequence of length 'limit'.
    """
    data = np.array(data)
    current_len = len(data)

    # If current length meets or exceeds limit, return truncated copy
    if current_len >= limit:
        return data[:limit].copy()

    # Number of points to extrapolate
    missing = limit - current_len

    # Estimate trend from the last two points (high weight)
    if current_len >= 2:
        delta = data[-1] - data[-2]  # Recent trend
    elif current_len == 1:
        delta = 0.0  # No trend if only one point
    else:
        raise ValueError("Input data must have at least one element")

    # Start extrapolation from the last observed value
    extrapolated = []
    current = data[-1]

    for i in range(1, missing + 1):
        # Apply trend with exponential decay to avoid aggressive long-term projection
        # Damping factor (e.g., 0.95^i) reduces trend influence over time
        damped_trend = delta * (0.95 ** i)
        current = current + damped_trend
        extrapolated.append(current)

    # Combine original data and extrapolated values
    extended = np.concatenate([data, np.array(extrapolated)])

    return extended

def plot_picture(files, datalen, ax):
    limit = 1000
    for filename in files:
        data_file = get_data(files[filename])
        x = []
        y = []
        for key in data_file:
            danow = data_file[key][choiceall[choice]]
            xnow = data_file[key][0]
            x.append(xnow)
            y.append(float(danow))


        if len(y)>=limit:
            if filename == "MSAN":
                ax.plot(y, label="MSAN")
            else:
                ax.plot(y[:limit], label= filename)
        else:
            if filename=="MSAN":
                y = data_deal(y, limit) # Smooth curve
                y = y
                x1 = moving_average(y, 10)
                x1[0:5] = y[0:5]
                x1[-10:] = y[-10:]
                y = x1
            else:
                y = y
                #y = data_deal(y, limit)  # Smooth curve
            if filename == "MSAN":
                ax.plot(y, label="MSAN")
            else:
                ax.plot(y, label=filename)

choiceall = {"Get_resultnum":3, "MeanReward":4, "MeanLength":5, "MeanLoss":6, "MeanQValue":7, "TimeDelta":8}
choicename = {"Get_resultnum": "Get_resultnum", "MeanReward": "Reward Mean", "MeanLength":"Trajectory Mean", "MeanLoss":6, "MeanQValue":"MeanQValue", "TimeDelta":"TimeDelta"}
choices = ["MeanReward", "MeanLength"]
choice = "MeanLength"
path = "./Env"
#map_file = ["S.png", "M.png", "H.png"]
#path_map = "./TTE/output/"
envlogs = ["S", "M", "H"]
envlogs_label = ["E1", "E2", "E3"]
datalens = ['10', '100']
files_all = []
fig, ax = plt.subplots(3, 4, figsize=(15, 8), dpi=60)
for ci, choice in enumerate(choices):
    for ai, envlog in enumerate(envlogs):
        for li, datalen in enumerate(datalens):
            files = get_file(path, envlog, datalen)
            files_all.append(files)
            plot_picture(files, datalen, ax[ai][li*2+ci])
            ax[ai][li*2+ci].set_ylabel(choicename[choice], fontsize=14)
            ax[ai][li*2+ci].set_xlabel("epochs-"+envlogs_label[ai]+"-"+str(datalens[li]), fontsize=14)

handles, labels = [], []
for ai, axx in enumerate(ax.flat):
    if ai <1:
        for h, l in zip(*axx.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
# fig.legend(handles, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.005))
plt.tight_layout()
# plt.subplots_adjust(top=0.956,
# bottom=0.084,
# left=0.046,
# right=0.986,
# hspace=0.374,
# wspace=0.292)
# ax333 = plt.axes((0.1, 0.75, 0.1, 0.2))
# ax333.set_xticks([])
# ax333.set_yticks([])
# ax333.spines['top'].set_visible(False)
# ax333.spines['right'].set_visible(False)
# ax333.spines['bottom'].set_visible(False)
# ax333.spines['left'].set_visible(False)
# ax333.imshow(Image.open(path_map+map_file[0]))
# ax333.set_xlabel("E1")
#
# #
# ax333 = plt.axes((0.071, 0.46, 0.16, 0.16))
# ax333.set_xticks([])
# ax333.set_yticks([])
# ax333.spines['top'].set_visible(False)
# ax333.spines['right'].set_visible(False)
# ax333.spines['bottom'].set_visible(False)
# ax333.spines['left'].set_visible(False)
# ax333.imshow(Image.open(path_map+map_file[1]))
# ax333.set_xlabel("E2", labelpad=-2)
# #
# ax333 = plt.axes((0.071, 0.13, 0.16, 0.16))
# ax333.set_xticks([])
# ax333.set_yticks([])
# ax333.spines['top'].set_visible(False)
# ax333.spines['right'].set_visible(False)
# ax333.spines['bottom'].set_visible(False)
# ax333.spines['left'].set_visible(False)
# ax333.imshow(Image.open(path_map+map_file[2]))
# ax333.set_xlabel("E3")

plt.show()