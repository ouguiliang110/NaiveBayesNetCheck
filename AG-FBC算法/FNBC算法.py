import math
import random
from minepy import MINE
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
X=np.loadtxt('../TAN算法/数据集/[023]sonar(0-1).txt')
m= X.shape[1] - 1
n = X.shape[0]

vector_data = X[:, :-1]
# 提取label类别
label_data = X[:, -1]

SetClass = set(X[:, m])
SetClass = list(map(int, SetClass))
# print(SetClass)
K = len(SetClass)  # 类标记数量

array1 = np.zeros(shape = (0, n))
for n in range(0, m):
    k = 10
    d1 = pd.cut(vector_data[:, n], k, labels = range(k))
    array1 = np.vstack((array1, d1))
array1 = np.vstack((array1, label_data))
X = array1.T
print(X)


 # 合并同类样本，顺序排列
newarray = [np.zeros(shape = [0, m + 1])] * 20
for i in X:
    for j in SetClass:
        if i[m] == j:
            newarray[j] = np.vstack((newarray[j], i))

NewArray = np.zeros(shape = [0, m + 1])
for i in SetClass:
    NewArray = np.vstack((NewArray, newarray[i]))
HiddenMatrix = NewArray
'''
#对类进行排序，即同类归类到一个中，方便比较
array1 = np.zeros(shape = (0, n))
for n in range(0, m):
    k = 25
    d1 = pd.cut(vector_data[:, n], k, labels = range(k))
    array1 = np.vstack((array1, d1))
array1 = np.vstack((array1, label_data))
X1 = array1.T
print(X1)
array = X1
'''

NumClass = [0] * K
# 初始化U
p = 0
for i in HiddenMatrix:
    for j in range(0, K):
        if i[m] == SetClass[j]:
            NumClass[j] = NumClass[j] + 1
    p = p + 1
# print(NumClass)

arr_train = []
arr_test = []

for k in range(0, 20):
    train = []
    trainNum = 0
    test = []
    testNum = 0
    for i in range(0, K):
        train.append(int(NumClass[i] * 0.7))
        trainNum += int(NumClass[i] * 0.7)

        test.append(NumClass[i] - train[i])
        testNum += NumClass[i] - train[i]

    train_index = []
    test_index = []
    for i in range(0, K):
        idx = np.random.choice(np.arange(NumClass[i]), size = train[i], replace = False)
        train_index.append(np.array(idx))
        # print(train_index)
        # val_index.append(np.random.choice(np.delete(np.arange(NumClass[i]), train_index[i]), size = val[i], replace = False))
        test_index.append(np.delete(np.arange(NumClass[i]), train_index[i]))
        # print(test_index)
    dividX = []
    p2 = 0
    for i in range(0, K):
        dividX.append(HiddenMatrix[p2:p2 + NumClass[i], :])
        p2 = p2 + NumClass[i]

    trainSet = []
    for i in range(0, K):
        trainSet.append(dividX[i][train_index[i], :])
    TrainSet = np.zeros((0, m + 1))
    for i in range(0, K):
        TrainSet = np.vstack((TrainSet, trainSet[i]))
    # print(TrainSet)
    Y = TrainSet[:, m]
    # print(Y)
    TrainSet = np.delete(TrainSet, m, axis = 1)
    for i in range(0, K):
        trainSet[i] = np.delete(trainSet[i], m, axis = 1)

    testSet = []
    for i in range(0, K):
        testSet.append(np.delete(dividX[i][test_index[i], :], m, axis = 1))
    # print(testSet)
    # print(testSet)

    # print(valSet)
    clf1 = GaussianNB()
    clf1.fit(TrainSet, Y)

    correct = 0
    for i in range(0, K):
        C = clf1.predict(testSet[i])
        # print(C)
        # print(SetClass[i])
        correct += sum(C == SetClass[i])
        # print(sum(C == SetClass[i]))
        # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
    testacc = correct / testNum
    arr_test.append(testacc)
    # print("test accuracy:{:.2%}".format(testacc))
    # print("---------------------------")
    correct1 = 0
    for i in range(0, K):
        C = clf1.predict(trainSet[i])
        # print(C)
        # print(SetClass[i])
        correct1 += sum(C == SetClass[i])
        # print(sum(C == SetClass[i]))
        # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
    arr_train.append(correct1 / trainNum)
    # print("train accuracy:{:.2 %}".format(trainacc))

# 求标准差
##print(arr_train)
print("测试精度", arr_test)
arr_mean = np.mean(arr_train)
arr_std = np.std(arr_train, ddof = 1)

arr_mean1 = np.mean(arr_test)
arr_std1 = np.std(arr_test, ddof = 1)

print("训练集平均标准差", arr_mean, arr_std)
print("测试集平均标准差", arr_mean1, arr_std1)