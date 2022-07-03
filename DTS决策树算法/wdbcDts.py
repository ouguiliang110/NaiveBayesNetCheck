import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import pandas as pd
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
X = np.loadtxt('6wdbc.txt',delimiter=',',skiprows = 1)
# 其中有97
m = X.shape[1] - 1  # 属性数量
#print(m)
n = X.shape[0]  # 样本数目
Y = X[:, m]
#print(Y)
print(X)



