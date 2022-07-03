import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from collections import Counter, defaultdict
from minepy import MINE
import numpy as np
import pandas as pd
import operator
import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report
#from sklearn.metrics import roc_curve
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
#from sklearn.externals import joblib


X = np.loadtxt('../数据集/机器学习作业数据集.txt')
m = X.shape[1] - 1  # 属性数量
print(m)
n = X.shape[0]  # 样本数目 其中第一类1219，第二类683

vector_data = X[:, :-1]
# 提取label类别
label_data = X[:, -1]

# data = pd.DataFrame(vector_data)
array1 = np.zeros(shape = (0, n))

for n in range(0,m):
    k = 10
    d1 = pd.cut(vector_data[:,n], k, labels = range(k))
    array1 = np.vstack((array1,d1))
array1=np.vstack((array1,label_data))
X1=array1.T
print(X1)
X=X1
'''
p1=0
for i in vector_data:
    k = 5
    d1 = pd.cut(i, k, labels = range(k))
    d1 = np.append(d1, label_data[p1])
    p1 += 1
    array = np.vstack((array, d1))

X=array
print(X)
'''

vector_data = X[:, :-1]
# 提取label类别
label_data = X[:, -1]

SetClass = set(X[:, m])
SetClass = list(map(int, SetClass))
print(SetClass)
K = len(SetClass)  # 类标记数量

NumClass = [0] * K
# 初始化U
p = 0
for i in X:
    for j in range(0, K):
        if i[m] == SetClass[j]:
            NumClass[j] = NumClass[j] + 1
    p = p + 1

#初始化
p_prior = {}  # 先验概率
JointPro = defaultdict(float)
Marginal = defaultdict(float)

n_samples = label_data.shape[0]  # 计算样本数
# 统计不同类别的样本数，并存入字典，key为类别，value为样本数
# Counter类的目的是用来跟踪值出现的次数。以字典的键值对形式存储，其中元素作为key，其计数作为value。
dict_label = Counter(label_data)
K = len(dict_label)

smooth=1
for key, val in dict_label.items():
    # 计算先验概率
    p_prior[key] = (val + smooth / K) / (n_samples + smooth)

nums_j = defaultdict(int)
nums_m = defaultdict(int)
for arr in X:
    nums_j[tuple(arr)] += 1
    JointPro[tuple(arr)]=0
ClassNums=[0]*K

for key,val in nums_j.items():
    ClassNums[int(key[m]-1)]+=1

for key,val in nums_j.items():
    JointPro[key]=val/NumClass[int(key[m]-1)]

AllMarginal=[]

for i in range(0,m):
    vector_x=vector_data[:,i]
    nums_sx=np.unique(vector_x)
    nums_vd = defaultdict(int)
    nums_pro=defaultdict(int)
    for xd,y in zip(vector_x,label_data):
        nums_vd[(xd,y)]+=1
    for key,val in nums_vd.items():
        nums_pro[(key[0],key[1])]=val/NumClass[int(key[1]-1)]
    AllMarginal.append(nums_pro)

#生成模型矩阵
getAllMetric=[]
metric=[]
tag=0
for key,val in JointPro.items():
    temp = []
    for j in range(0, len(key) - 1):
        temp.append(AllMarginal[j][(key[j], key[m])])
    temp.append(val)
    metric.append(temp)
metric=np.array(metric)
print(metric)
p1=0
#print(metric[p1:p1+NumClass[0],:])
for i in range(0,K):
    getAllMetric.append(metric[p1:p1+ClassNums[i],:])
    p1+=ClassNums[i]

#print(getAllMetric[1][:, :-1])
#print(getAllMetric[1][:, -1])
bpAll=[]
'''
for i in range(0,K):
    scaler=StandardScaler()
    scaler.fit(getAllMetric[i][:,:-1])
    getAllMetric[i][:,:-1]=scaler.transform(getAllMetric[i][:,:-1])
'''



#print(getAllMetric[1][:,:-1])
#print(getAllMetric[1][:, -1])
for i in range(0,K):
  bp=MLPRegressor(hidden_layer_sizes=(1000), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=2000,tol=1e-4)
  bp.fit(getAllMetric[i][:, :-1],getAllMetric[i][:, -1])
  bpAll.append(bp)
  #print(bpAll)
getvalue=[0]*K

print(bpAll[0].predict(getAllMetric[0][:, :-1]))
print(bpAll[1].predict(getAllMetric[1][:, :-1]))
for i in range(0,K):
    indexNew=[]
    for col in getAllMetric[i][:, :-1]:
        theNew = []
        theNew.append(bpAll[i].predict([col])[0] * p_prior[i+1])
        for j in range(0,K):
            if i!=j:
               theNew.append(bpAll[j].predict([col])[0] * p_prior[j+1])
        #print(theNew)
        indexNew.append(theNew.index(max(theNew)))
        if theNew.index(max(theNew))==0:
           getvalue[i]+=1
    print(indexNew)
print(sum(getvalue)/sum(ClassNums))
