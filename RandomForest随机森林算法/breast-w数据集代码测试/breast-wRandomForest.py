import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier


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
X = np.loadtxt('[004]bd(0-1).txt')
# 其中有97
m = 30  # 属性数量
n = 214  # 样本数目 其中第一类212，第二类357    148  64   250 107
K = 6  # 类标记数量
# 去掉类标记
Class1=0
Class2=0
for i in X:
    if i[30] == 1:
        Class1 = Class1 + 1
    elif i[30] == 2:
        Class2 = Class2 + 1
train1=int(Class1*0.7)
test1=Class1-train1
train2=int(Class2*0.7)
test2=Class2-train2
Y = X[:, 9]
print(X.shape)
#print(Y)

'''
array1 = np.delete(X,9, axis = 1)
array = np.zeros(shape = (0, 9))
for i in array1:
    k = 20
    d1 = pd.cut(i, k, labels = range(k))
    array = np.vstack((array, d1))
Y=Y.reshape(n,1)
X=np.concatenate((array,Y),axis=1)
'''


X1 = X[0:212, :]
X2 = X[212:569, :]

acc=[]
for i in range(0, 20):
    # 随机抽取样本训练集和测试集样本
    idx = np.random.choice(np.arange(Class1), size = train1, replace = False)
    train_index1 = np.array(idx)
    test_index1 = np.delete(np.arange(Class1), train_index1)

    idx1 = np.random.choice(np.arange(Class2), size = train2, replace = False)
    train_index2 = np.array(idx1)
    test_index2 = np.delete(np.arange(Class2), train_index2)

    # 取训练集和测试集7：3比例
    Data1 = X1[train_index1, :]
    Data2 = X2[train_index2, :]


    trainingSet = np.vstack((Data1, Data2))
    trainingSetX = np.delete(trainingSet, 30, axis = 1)
    trainingSetY = trainingSet[:, 30]


    '''
    Data1 = np.delete(Data1, 9, axis = 1)
    Data2 = np.delete(Data2, 9, axis = 1)
    Data3= np.delete(Data3, 9, axis = 1)
    Data4 = np.delete(Data4, 9, axis = 1)
    Data5 = np.delete(Data5, 9, axis = 1)
    Data6 = np.delete(Data6, 9, axis = 1)
    '''



    testSet1 = np.delete(X1[test_index1, :], 30, axis = 1)
    testSet2 = np.delete(X2[test_index2, :], 30, axis = 1)
    # testSet = np.vstack((testSet1, testSet2))
    # print(testSet)
    # 统计正确数量和计算准确率
    clf = RandomForestClassifier()
    clf.fit(trainingSetX, trainingSetY)
    C1 = clf.predict(testSet1)
    add = sum(C1 == 1)
    print(add)
    C2 = clf.predict(testSet2)
    add1 = sum(C2 == 2)
    print(add1)

    acc.append((add + add1) / (test1+test2))
    print("accuracy:{:.2%}".format((add + add1) / (test1+test2)))
arr_mean = np.mean(acc)
# 求方差
arr_var = np.var(acc)
arr_std = np.std(acc, ddof = 1)
print(arr_mean,arr_var,arr_std)
'''
# 求各类对应属性的均值和方差
Mean1 = np.mean(Data1, axis = 0)
Mean2 = np.mean(Data2, axis = 0)
var1 = np.var(Data1, axis = 0)
var2 = np.var(Data2, axis = 0)
'''
