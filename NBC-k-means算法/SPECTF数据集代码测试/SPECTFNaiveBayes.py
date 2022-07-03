import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
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

X = np.loadtxt('[024]SPECTF(0-1).txt')
X1=np.delete(X,X.shape[1]-1,axis = 1)
df=pd.DataFrame(X1)
sns.pairplot(df)
plt.show()
# 其中有97
m = 44  # 属性数量
n = 267  # 样本数目
K = 2  # 类标记数量

# 去掉类标记
X = np.delete(X, 44, axis = 1)

# 取训练集和测试集5；2：3比例
trainSet1=X[0:106, :]
trainSet2=X[212:239,:]
valSet1=X[106:148,:]
valSet2=X[239:250,:]
# print(trainingSet)
testSet1=X[148:212, :]
testSet2=X[250:267, :]
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
Mean2 = np.mean(trainSet2, axis = 0)
var1 = np.var(trainSet1, axis = 0)
var2 = np.var(trainSet2, axis = 0)

# 先求P(C) 根据机器学习课本中的拉普拉斯修正法
Pro1 =106 / 133
Pro2 = 27 / 133
# print(Pro1)
# print(Pro2)

# 本次代码主要内容是这个，求P(Ai|C)

# 统计正确数量和计算准确率
add = 0
for i in range(0, 106):
    sum = 1
    for j in range(0, m):
        sum *= getPro(trainSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, m):
        sum1 *= getPro(trainSet1[i][j], Mean2[j], var2[j])
    if Pro1*sum >= Pro2*sum1:
        add += 1
    elif Pro1*sum < Pro2*sum1:
        add += 0
print("第一类正确数量(总数65)：")
print(add)
add1=0
for i in range(0, 27):
    sum = 1
    for j in range(0, m):
        sum *= getPro(trainSet2[i][j], Mean2[j], var2[j])
    sum1 = 1
    for j in range(0, m):
        sum1 *= getPro(trainSet2[i][j], Mean1[j], var1[j])
    if Pro2*sum >= Pro1*sum1:
        add1 += 1
    elif Pro2*sum < Pro1*sum1:
        add1 += 0
    print("-------------")
    print(sum)
    print(sum1)
    print("---------------")
print("第二类正确数量(总数20)：")
print(add1)
#准确率
print("accuracy:{:.2%}".format((add+add1)/133))

