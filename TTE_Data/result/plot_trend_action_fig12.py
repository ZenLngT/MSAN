import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import pandas as pd
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import random
import matplotlib
# 先导入相关包
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from pylab import mpl
#matplotlib.rcParams['font.sans-serif'] = ['STSong']
#plt.rcParams['axes.unicode_minus'] = False    # 显示负号

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

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def data_deal(data, limit):
    if len(data)>0:
        x1 = moving_average(data, 50)
        return x1[25:-25]
    else:
        return data

def plot_picture(ax, files, flag):
    limit = 10000 #长度
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    data_all = []
    for filename in files:
        data_file = get_data(filename)
        #data_file = get_data(files[filename])
        choicenow = choiceall[choice]
        for coi in range(choicenow[0], choicenow[1]+1):
            x = []
            y = []
            for key in data_file:
                danow = data_file[key][coi]
                xnow = data_file[key][0]
                x.append(xnow)
                y.append(float(danow))
            data_all.append(y)
    for labi, y  in enumerate(data_all):
        ax.plot(y, alpha=0.1, color=colors[labi])
        y = data_deal(y, limit)
        ax.plot(y, label=choicename["Action"][labi], color=colors[labi])
        # plt.plot(y, label=filename)
        labi += 1

choiceall = {"Get_resultnum":3, "MeanReward":4, "MeanLength":5, "MeanLoss":[15,19], "MeanQValue":[21,25], "TimeDelta":8, "Action":[9,13]}
choicename = {"Get_resultnum": "得解率", "MeanReward": "MeanReward", "MeanLength":"平均探索长度", "MeanLoss": 6, "MeanQValue":7, "TimeDelta":8, "Action":["sub-DDPG", "sub-PPO","sub-PG","sub-DQN","sub-DDQN", "decison-Master"]}
choice = "Action"
path = ["./loss/E1_100_100.log"]
#envlog = "S"  # 需要和path对应
#datalen = '100'  # 需要和path对应
flag = 0
#files_all = get_file(path, envlog, datalen)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_picture(ax, path, flag)
plt.rcParams.update({'font.size': 12})
plt.legend(loc='right')  # label 位置左上
plt.tick_params(labelsize=12)  # label的字体大小，label表示图中的DDPG等
plt.ylabel("Execute Number", fontsize=14)
plt.xlabel("epochs", fontsize=15)  # fontsize=x轴说明字体大小
plt.tight_layout()
axins = inset_axes(ax, width="40%", height="40%", loc='center',
                   bbox_to_anchor=(-0.25, 0.1, 1.1, 1.1),
                   bbox_transform=ax.transAxes)
plot_picture(axins, path, flag)
# 调整子坐标系的显示范围
axins.set_xlim(0, 6000)
axins.set_ylim(0, 17.5)
# 画两条线
xy = (0, 0)
xy2 = (0, 0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (6000,0)
xy2 = (6000, 0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)
#plt.savefig('../output/picture/Execute_Experiment'+".png", dpi=300)
plt.show()