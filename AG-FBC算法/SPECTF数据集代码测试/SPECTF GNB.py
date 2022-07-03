import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
import random


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro

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

X = np.loadtxt('[024]SPECTF(0-1).txt')
# 其中有97
m = 44  # 属性数量
n = 267  # 样本数目
K = 2  # 类标记数量
Y=X[:,m]
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


clf.fit(X,Y)
C1=clf.predict(trainSet1)
add=sum(C1==1)
print(add)
C2=clf.predict(trainSet2)
add1=sum(C2==2)
print(add1)
print("accuracy:{:.2%}".format((add+add1)/133))
