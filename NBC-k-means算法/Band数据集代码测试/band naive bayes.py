import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    a=1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro=1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro

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

X = np.loadtxt('[008]band(0-1).txt')

NewArray = np.delete(X, X.shape[1]-1, axis = 1)
df=pd.DataFrame(NewArray)
sns.pairplot(df)
plt.show()

# 其中有97
m = X.shape[1]-1  # 属性数量
n = X.shape[0]  # 样本数目
K = 2  # 类标记数量

# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

train1 = int(Class1 * 0.7)
test1 = Class1 - train1

train2 = int(Class2 * 0.7)
test2 = Class2 - train2


X1 = X[0:Class1, :]
X2 = X[Class1:Class1 + Class2, :]
acc=[]
for i in range(0,20):
    # 随机抽取样本训练集和测试集样本
    idx = np.random.choice(np.arange(Class1), size = train1, replace = False)
    train_index1 = np.array(idx)
    test_index1 = np.delete(np.arange(Class1), train_index1)

    idx1 = np.random.choice(np.arange(Class2), size = train2, replace = False)
    train_index2 = np.array(idx1)
    test_index2 = np.delete(np.arange(Class2), train_index2)

    Data1 = X1[train_index1, :]
    Data2 = X2[train_index2, :]

    testSet1 = np.delete(X1[test_index1, :], m, axis = 1)
    testSet2 = np.delete(X2[test_index2, :], m, axis = 1)
    trainSet1=np.delete(Data1,m,axis = 1)
    trainSet2=np.delete(Data2,m,axis = 1)

    # 求各类对应属性的均值和方差
    Mean1 = np.mean(trainSet1, axis = 0)
    Mean2 = np.mean(trainSet2, axis = 0)
    print(Mean2)
    var1 = np.var(trainSet1, axis = 0)
    var2 = np.var(trainSet2, axis = 0)
    # 先求P(C)
    Pro1 = (train1 + 1) / (train1+train2 + 1)
    Pro2 = (train2 + 1) / (train1+train2 + 1)
    add = 0
    for i in range(0, train1):
        sum = 1
        for j in range(0, m):
            sum *= getPro(trainSet1[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(trainSet1[i][j], Mean2[j], var2[j])
        if Pro1 * sum >= Pro2 * sum1:
            add += 1
        elif Pro1 * sum < Pro2 * sum1:
            add += 0
    print("第一类正确数量(总数27)：")
    print(add)
    add1 = 0
    for i in range(0, train2):
        sum = 1
        for j in range(0, m):
            sum *= getPro(trainSet2[i][j], Mean2[j], var2[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(trainSet2[i][j], Mean1[j], var1[j])
        if Pro2 * sum >= Pro1 * sum1:
            add1 += 1
        elif Pro2 * sum < Pro1 * sum1:
            add1 += 0
    print("第二类正确数量(总数34)：")
    print(add1)
    acc.append((add+add1)/(train1+train2))
    print("accuracy:{:.2%}".format((add + add1) / (train1+train2)))
arr_mean = np.mean(acc)
# 求方差
arr_var = np.var(acc)
arr_std = np.std(acc, ddof = 1)
print(arr_mean, arr_var, arr_std)

