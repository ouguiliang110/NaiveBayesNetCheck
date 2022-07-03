import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from collections import Counter, defaultdict
from minepy import MINE
import pandas as pd
import operator
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from sko.PSO import PSO

plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

length=20
weightmatrix= [1.8391859347929786, 2.116253190349208, 2.6324629338870726, 2.330775415204829, 2.0215834441984233, 2.1270941492967405, 2.2339503167233987, 2.34440042079805, 2.3414885010754882, 2.3190536913152267, 2.3174536136032885, 2.317488502654246, 2.3179544168086374, 2.3176618561746706, 2.317018666616727, 2.317183288185578, 2.3177587914876157, 2.3177187018744803, 2.31890343496444125, 2.318902292389116]
ling=[0.28064581224334884, 0.27852031986840236, 0.26954365668597733, 0.25609166867135675, 0.2763943398983916, 0.270875294257183, 0.2782157169214163, 0.2824569092117132, 0.29247151412452026, 0.29171411967260634, 0.30208972699479744, 0.30208972699479744, 0.30208972699479744, 0.30208972699479744,0.30208972699479744, 0.30208972699479744, 0.30208972699479744, 0.30208972699479744, 0.30208972699479744, 0.30208972699479744]
print(len(ling))
ling_std1=np.array(weightmatrix)+np.array(ling)
ling_std2=np.array(weightmatrix)-np.array(ling)
Good_test_Accuracy=np.array(weightmatrix)
ling_mean=np.mean(Good_test_Accuracy,axis = 0)
#print(ling_mean)
plt.plot(range(length), weightmatrix, '-')
plt.fill_between(range(length), ling_std1, ling_std2, alpha = 0.3)
plt.xlabel('迭代次数',fontsize=12)
plt.ylim((1.4,3.0))
plt.ylabel('权重w',fontsize=12)
plt.show()