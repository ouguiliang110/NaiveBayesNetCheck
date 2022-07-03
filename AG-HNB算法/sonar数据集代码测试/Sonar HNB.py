import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

X = np.loadtxt('[023]sonar(0-1).txt')
Y=X[:,60]
X = np.delete(X, 60, axis = 1)
# 其中有97
m = 60  # 属性数量
n = 208  # 样本数目
K = 2  # 类标记数量

data = pd.DataFrame(X)
array = np.zeros(shape = (0, 60))
for i in X:
    k = 8
    d1 = pd.cut(i, k, labels = range(k))
    array = np.vstack((array, d1))
print(array)


def getPAiAj(Data1,Data2,X):
    for i in range(0,n):
def getI():

def getWij():

X1 = array[0:97, :]
X2 = array[97:208, :]
# 随机抽取样本训练集和测试集样本
for i in range(0,10):
    idx = np.random.choice(np.arange(97), size = 70, replace = False)
    train_index1 = np.array(idx)
    test_index1 = np.delete(np.arange(97), train_index1)
    idx1 = np.random.choice(np.arange(111), size = 77, replace = False)
    train_index2 = np.array(idx1)
    test_index2 = np.delete(np.arange(111), train_index2)
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