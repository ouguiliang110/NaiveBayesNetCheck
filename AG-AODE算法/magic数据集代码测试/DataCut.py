import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

X = np.loadtxt('[017]magic(0-1).txt')
Y=X[:,10]
X = np.delete(X, 10, axis = 1)
m = 10  # 属性数量
n = 1902  # 样本数目 其中第一类1219，第二类683
K = 2  # 类标记数量

data = pd.DataFrame(X)
array = np.zeros(shape = (0, 10))
for i in X:
    k = 5
    d1 = pd.cut(i, k, labels = range(k))
    array = np.vstack((array, d1))
print(array)

X1 = array[0:1219, :]
X2 = array[1219:1902, :]
# 随机抽取样本训练集和测试集样本
for i in range(0,10):
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
    clf = MultinomialNB()
    clf.fit(array, Y)
    C1 = clf.predict(Data1)
    add = sum(C1 == 1)
    print(add)
    C2 = clf.predict(Data2)
    add1 = sum(C2 == 2)
    print(add1)
    print("accuracy:{:.2%}".format((add + add1) / 1331))