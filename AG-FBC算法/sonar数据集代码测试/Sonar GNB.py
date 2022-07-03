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

clf = GaussianNB()
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

Y=X[:,60]
print(Y)
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

clf.fit(X,Y)
C1=clf.predict(testSet1)
add=sum(C1==1)
print(add)
C2=clf.predict(testSet2)
add1=sum(C2==2)
print(add1)
print("accuracy:{:.2%}".format((add+add1)/61))