import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
import random


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro

def getK(Data, X):
    add = 0
    n = Data.shape[0]
    for i in range(0, n):
        add += 1 / math.sqrt(2 * math.pi * n) * math.exp(-(np.sum(np.square(Data[i] - X)) * n / 2))
    return add

'''
def CountP1(test):
    sum=1
    for i in range(0,60):
       sum*=getPro(test[i],
def CountP2(test):
    sum=1
    for i in  range(0,60):
        sum*=getPro(())
'''
clf = GaussianNB()
X = np.loadtxt('[017]magic(0-1).txt')
# 其中有97
m = 10  # 属性数量
n = 1902  # 样本数目 其中第一类1219，第二类683
K = 2  # 类标记数量
# 去掉类标记
Y=X[:,10]
X = np.delete(X, 10, axis = 1)
X1=X[0:1219,:]
X2=X[1219:1902,:]


for i in range(0,10):
    # 随机抽取样本训练集和测试集样本
    idx = np.random.choice(np.arange(1219), size = 853, replace = False)
    train_index1 = np.array(idx)
    test_index1 = np.delete(np.arange(1219), train_index1)
    idx1 = np.random.choice(np.arange(683), size = 478, replace = False)
    train_index2 = np.array(idx1)
    test_index2 = np.delete(np.arange(683), train_index2)
    # 取训练集和测试集7：3比例
    Data1 = X1[train_index1, :]
    Data2 = X2[train_index2, :]
    trainingSet = np.vstack((Data1, Data2))
    # print(trainingSet)
    testSet1 = X1[test_index1, :]
    testSet2 = X2[test_index2, :]
    # testSet = np.vstack((testSet1, testSet2))
    # print(testSet)
    # 统计正确数量和计算准确率
    clf.fit(X, Y)
    C1 = clf.predict(Data1)
    add = sum(C1 == 1)
    print(add)
    C2 = clf.predict(Data2)
    add1 = sum(C2 == 2)
    print(add1)
    print("accuracy:{:.2%}".format((add + add1) / 1331))

'''
# 求各类对应属性的均值和方差
Mean1 = np.mean(Data1, axis = 0)
Mean2 = np.mean(Data2, axis = 0)
var1 = np.var(Data1, axis = 0)
var2 = np.var(Data2, axis = 0)
'''



