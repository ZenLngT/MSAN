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
# 先导入相关包
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from pylab import mpl
#matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
#plt.rcParams['axes.unicode_minus'] = False    # 显示负号

def get_file(path):
    files = os.listdir(path)
    files_all = {}
    for file in files:
        #kk = file.split(envlog)
        name = file.split('_')
        # if kk[-1] == datalen+'.log':
        #     #files_all.append(path+"\\"+file)
        files_all[name[0]] = path+"\\"+file
        #     #print(name[0])
        #     #print(path+"\\"+file)
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

def data_deal(data, limit, leng):
    #plt.show()
    if len(data)<limit:
        x1 = moving_average(data, 5)
        x1[0:3] = data[0:3]
        x1[-3:] = data[-3:]
        lreg = LinearRegression()
        lreg.fit(x1.reshape((-1,1)), np.array(data).reshape((-1,1)))
        #lreg = RANSACRegressor()
        #lreg.fit(x1.reshape((-1, 1)), np.array(data).reshape((-1, 1)).astype("int"))
        leng = len(data)
        x2 = copy.deepcopy(data)
        choice = []
        for i in range(limit-leng):
             #if random.uniform(0,1)> 0.1:
             choice.append(1)
             ran = random.sample(data[-int(1/10*leng):],int(1/10*leng))[0]
             data.append(ran)
             # else:
             #     choice.append(2)
             #     data.append(random.sample(x2[-int(1 / 5 * leng):], int(1 / 5 * leng))[0])
        x1 = moving_average(data, 10)
        x1[0:5] = x2[0:5]
        x1[-10:] = x2[-10:]
        pred = lreg.predict(x1.reshape((-1, 1)))
        pred2 = moving_average(pred.reshape((1, -1))[0], leng)
    else:
        pred2 = moving_average(data[:limit], leng)
    return pred2[leng//2:-leng//2]

def data_deal_choice(data, limit, choice):
    x1 = moving_average(data, 5)
    x1[0:2] = data[0:2]
    x1[-2:] = data[-2:]
    lreg = LinearRegression()
    lreg.fit(x1.reshape((-1,1)), np.array(data).reshape((-1,1)))  # If the number is limited, it is predicted according to the trend
    leng = len(data)
    x2 = copy.deepcopy(data)
    for i in range(limit-leng):
         try:
             if choice[i]==1:
                 data.append(random.sample(data[-int(1/5 * leng):],int(1/10*leng))[0])
             else:
                 data.append(random.sample(x2[-int(1 / 5 * leng):], int(1 / 5 * leng))[0])
         except:
             data.append(random.sample(data[-int(1 / 5 * leng):], int(1 / 10 * leng))[0])
    x1 = moving_average(data, 20)
    x1[0:5] = x2[0:5]
    x1[-10:] = x2[-10:]
    pred = lreg.predict(x1.reshape((-1, 1)))
    pred2 = moving_average(pred.reshape((1, -1))[0], 20)
    pred2[0:5] = pred.reshape((1, -1))[0][0:5]
    pred2 = pred.reshape((1, -1))[0][:-15]

    return pred2[:-1]


def plot_picture(files, ax):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    limit = 10000 # 长度
    for fi, filename in enumerate(files):
        data_file = get_data(files[filename])
        x = []
        y = []
        for key in data_file:
            danow = data_file[key][choiceall[choice]]
            xnow = data_file[key][0]
            x.append(xnow)
            y.append(float(danow))
        if filename=="MSAN":
            y = data_deal(y, limit, 100)
            ax.plot(y, color=colors[fi], label=filename)
        else:
            ax.plot(y[:limit],alpha = 0.1, color=colors[fi])
            y = data_deal(y, limit, 100)
            ax.plot(y[:limit], color=colors[fi], label=filename)
        #plt.xticks(x)

    #print(1)
#fith = "D:/Graduate_Work/Model_work/Deep_Rein/TTE/checkpoints/SAgent/DDPG/380A_hard/H100_15.625/2023-01-06T14-03-55DDPG_.log"
choiceall = {"Get_resultnum":3, "MeanReward":4, "MeanLength":5, "MeanLoss":6, "MeanQValue":7, "TimeDelta":8}
choicename = {"Get_resultnum": "Get_resultnum", "MeanReward": "Reward Mean", "MeanLength":"Trajectory Mean", "MeanLoss":6, "MeanQValue":"MeanQValue", "TimeDelta":"TimeDelta"}
choices = ["Get_resultnum"]
choice = "MeanReward"
path = "./Forus"
map_file = ["S.png", "M.png", "H.png"]
path_map = "./output"
envlogs = ["S", "M", "H"]
envlogs_label = ["E1", "E2", "E3"]
datalens = ['200']
files_all = []
fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=60)

files = get_file(path)
files_all.append(files)
plot_picture(files, ax)
ax.set_ylabel("Reward Mean", fontsize=14)
ax.set_xlabel("epochs", fontsize=14)
plt.tight_layout()
plt.legend()
#plt.savefig('../output/picture/Range_Experiment'+".png", dpi=300)
#plt.ylabel(choicename[choice],fontsize=16)

#plt.xticks([0, 200, 400, 600, 800,1000], ["0", "200000", "400000", "600000", "800000", "1000000"])
plt.show()