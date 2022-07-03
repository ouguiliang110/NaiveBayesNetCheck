import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

X = np.loadtxt('[024]SPECTF(0-1).txt')
Y = X[:, 44]
X = np.delete(X, 44, axis = 1)
# 其中有97

# 其中有97
m = 44  # 属性数量
n = 267  # 样本数目
T = 3
K = 2  # 类标记数量
Class1 = 212
Class2 = 55

data = pd.DataFrame(X)
array = np.zeros(shape = (0, 44))
for i in X:
    k = 4
    d1 = pd.cut(i, k, labels = range(k))
    array = np.vstack((array, d1))
#print(array)

X1 = array[0:212, :]
X2 = array[212:267, :]

#print(X6)

# 随机抽取样本训练集和测试集样本
for i in range(0, 10):
    idx1 = np.random.choice(np.arange(212), size = 148, replace = False)
    train_index1 = np.array(idx1)
    test_index1 = np.delete(np.arange(212), train_index1)

    idx2 = np.random.choice(np.arange(55), size =38, replace = False)
    train_index2 = np.array(idx2)
    test_index2 = np.delete(np.arange(55), train_index2)

    #print(idx6)
    # 取训练集和测试集7：3比例
    trainSet1 = X1[train_index1, :]
    trainSet2 = X2[train_index2, :]

    #trainingSet = np.vstack((Data1, Data2))
    # print(trainingSet)
    testSet1 = X1[test_index1, :]
    testSet2 = X2[test_index2, :]

    # testSet = np.vstack((testSet1, testSet2))
    # print(testSet)
    # 统计正确数量和计算准确率
    clf = MultinomialNB()
    clf.fit(array, Y)
    C1 = clf.predict(trainSet1)
    add = sum(C1 == 1)
    #print(add)
    C2 = clf.predict(trainSet2)
    add1 = sum(C2 == 2)

    print("accuracy:{:.2%}".format((add + add1) / 186))
