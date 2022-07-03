import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
import random


# 连续型数据分类用正态分布公式
'''
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro
'''


def getK(Data, X):
    add = 0
    n = Data.shape[0]
    h=1/math.sqrt(n)
    for i in range(0, n):
        add += 1 / (h*math.sqrt(2 * math.pi)) * math.exp(-(np.sum(np.square(Data[i] - X))  / (2*h**2)))
    return add*1/n


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

X = np.loadtxt('[023]sonar(0-1).txt')
# 其中有97
m = 60  # 属性数量
n = 208  # 样本数目
K = 2  # 类标记数量

# 去掉类标记
X = np.delete(X, 60, axis = 1)

# 取训练集和测试机7：3比例
Data1 = X[0:70, :]
Data2 = X[97:174, :]
trainingSet = np.vstack((Data1, Data2))
# print
testSet1 = X[70:97, :]
testSet2 = X[174:208, :]
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

'''
# 求各类对应属性的均值和方差
Mean1 = np.mean(Data1, axis = 0)
Mean2 = np.mean(Data2, axis = 0)
var1 = np.var(Data1, axis = 0)
var2 = np.var(Data2, axis = 0)
'''

print()
# 本次代码主要内容是这个，求P(Ai|C)
# 求准确率
add = 0
for j in range(0, 27):
    K1 = getK(Data1, testSet1[j])
    K2 = getK(Data2, testSet1[j])
    print(K1/(K1+K2))
    if (K1 / (K1 + K2)) > 0.5:
        add += 1
    else:
        add += 0
print("第一类正确数量(总数27)：")
print(add)

add1 = 0
for j in range(0, 34):
    K1 = getK(Data1, testSet2[j])
    K2 = getK(Data2, testSet2[j])
    print(K2/(K1+K2))
    if (K2 / (K1 + K2)) > 0.5:
        add1 += 1
    else:
        add1 += 0
print("第二类正确数量(总数34)：")
print(add1)

print("accuracy:{:.2%}".format((add+add1)/61))
