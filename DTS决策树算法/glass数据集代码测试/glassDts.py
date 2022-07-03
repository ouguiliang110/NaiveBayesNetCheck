import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
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
X = np.loadtxt('[010]glass(0-1).txt')
# 其中有97
m = 9  # 属性数量
n = 214  # 样本数目 其中第一类1219，第二类683
K = 6  # 类标记数量
# 去掉类标记
Y = X[:, 9]
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


X1 = X[0:70, :]
X2 = X[70:146, :]
X3 = X[146:163, :]
X4 = X[163:192, :]
X5 = X[192:205, :]
X6 = X[205:214, :]
acc=[]
for i in range(0, 10):
    # 随机抽取样本训练集和测试集样本
    idx = np.random.choice(np.arange(70), size = 49, replace = False)
    train_index1 = np.array(idx)
    test_index1 = np.delete(np.arange(70), train_index1)

    idx1 = np.random.choice(np.arange(76), size = 53, replace = False)
    train_index2 = np.array(idx1)
    test_index2 = np.delete(np.arange(76), train_index2)

    idx2 = np.random.choice(np.arange(17), size = 12, replace = False)
    train_index3 = np.array(idx2)
    test_index3 = np.delete(np.arange(17), train_index3)

    idx3 = np.random.choice(np.arange(29), size = 20, replace = False)
    train_index4 = np.array(idx3)
    test_index4 = np.delete(np.arange(29), train_index4)

    idx4 = np.random.choice(np.arange(13), size = 9, replace = False)
    train_index5 = np.array(idx4)
    test_index5 = np.delete(np.arange(13), train_index5)

    idx5 = np.random.choice(np.arange(9), size = 6, replace = False)
    train_index6 = np.array(idx5)
    test_index6 = np.delete(np.arange(9), train_index6)
    # 取训练集和测试集7：3比例
    Data1 = X1[train_index1, :]
    Data2 = X2[train_index2, :]
    Data3 = X3[train_index3, :]
    Data4 = X4[train_index4, :]
    Data5 = X5[train_index5, :]
    Data6 = X6[train_index6, :]


    trainingSet = np.vstack((Data1, Data2, Data3, Data4, Data5, Data6))
    trainingSetX = np.delete(trainingSet, 9, axis = 1)

    trainingSetY = trainingSet[:, 9]



    Data1 = np.delete(Data1, 9, axis = 1)
    Data2 = np.delete(Data2, 9, axis = 1)
    Data3= np.delete(Data3, 9, axis = 1)
    Data4 = np.delete(Data4, 9, axis = 1)
    Data5 = np.delete(Data5, 9, axis = 1)
    Data6 = np.delete(Data6, 9, axis = 1)



    testSet1 = np.delete(X1[test_index1, :], 9, axis = 1)
    testSet2 = np.delete(X2[test_index2, :], 9, axis = 1)
    testSet3= np.delete(X3[test_index3, :], 9, axis = 1)
    testSet4 = np.delete(X4[test_index4, :],9, axis = 1)
    testSet5 = np.delete(X5[test_index5, :], 9, axis = 1)
    testSet6 = np.delete(X6[test_index6, :], 9, axis = 1)
    # testSet = np.vstack((testSet1, testSet2))
    # print(testSet)
    # 统计正确数量和计算准确率
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainingSetX, trainingSetY)
    C1 = clf.predict(Data1)
    add = sum(C1 == 1)
    print(add)
    C2 = clf.predict(Data2)
    add1 = sum(C2 == 2)
    print(add1)
    C3 = clf.predict(Data3)
    add2 = sum(C3 == 3)
    print(add2)
    C4 = clf.predict(Data4)
    add3 = sum(C4 == 4)
    print(add3)
    C5 = clf.predict(Data5)
    add4 = sum(C5 == 5)
    print(add4)
    C6 = clf.predict(Data6)
    add5 = sum(C6 == 6)
    print(add5)
    acc.append((add + add1+add2+add3+add4+add5) / 65)
    print("accuracy:{:.2%}".format((add + add1+add2+add3+add4+add5) / 65))
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
