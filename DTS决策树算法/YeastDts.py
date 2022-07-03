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
X = np.loadtxt('[030]yeast(0-1).txt')
print(X)
# 其中有97
m = X.shape[1] - 1  # 属性数量
n = X.shape[0]  # 样本数目
print(n,m)
Y = X[:, m]
# 去掉类标记
Class1 = 0
Class2 = 0
Class3 = 0
Class4 = 0
Class5 = 0
Class6 = 0
Class7 = 0
Class8 = 0
Class9 = 0
Class10 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1
    elif i[m] == 3:
        Class3 = Class3 + 1
    elif i[m] == 4:
        Class4 = Class4 + 1
    elif i[m] == 5:
        Class5 = Class5 + 1
    elif i[m] == 6:
        Class6 = Class6 + 1
    elif i[m] == 7:
        Class7 = Class7 + 1
    elif i[m] == 8:
        Class8 = Class8 + 1
    elif i[m] == 9:
        Class9 = Class9 + 1
    elif i[m] == 10:
        Class10 = Class10 + 1
print(Class4)
train1 = int(Class1 * 0.7)
test1 = Class1 - train1

train2 = int(Class2 * 0.7)
test2 = Class2 - train2

train3 = int(Class3 * 0.7)
test3 = Class3 - train3

train4 = int(Class4 * 0.7)
test4 = Class4 - train4

train5 = int(Class5 * 0.7)
test5 = Class5 - train5

train6 = int(Class6 * 0.7)
test6 = Class6 - train6

train7 = int(Class7 * 0.7)
test7 = Class7 - train7

train8 = int(Class8 * 0.7)
test8 = Class8 - train8

train9 = int(Class9 * 0.7)
test9 = Class9 - train9

train10 = int(Class10 * 0.7)
test10 = Class10 - train10
# print(Y)

'''
array1 = np.delete(X,m, axis = 1)
array = np.zeros(shape = (0, m))
for i in array1:
    k = 20
    d1 = pd.cut(i, k, labels = range(k))
    array = np.vstack((array, d1))
Y=Y.reshape(n,1)
X=np.concatenate((array,Y),axis=1)

'''




#print(X)
X1 = X[0:Class1, :]
X2 = X[Class1:Class1 + Class2, :]
X3 = X[Class1 + Class2:Class1 + Class2 + Class3, :]
X4 = X[Class1 + Class2 + Class3:Class1 + Class2 + Class3+Class4, :]
#print(X4)
X5 = X[Class1 + Class2 + Class3+Class4:Class1 + Class2 + Class3+Class4+Class5, :]
#print(X5)
X6 = X[Class1 + Class2 + Class3+Class4+Class5:Class1 + Class2 + Class3+Class4+Class5+Class6, :]
X7 = X[Class1 + Class2 + Class3+Class4+Class5+Class6:Class1 + Class2 + Class3+Class4+Class5+Class6+Class7, :]
X8 = X[Class1 + Class2 + Class3+Class4+Class5+Class6+Class7:Class1 + Class2 + Class3+Class4+Class5+Class6+Class7+Class8, :]
X9 = X[Class1 + Class2 + Class3+Class4+Class5+Class6+Class7+Class8:Class1 + Class2 + Class3+Class4+Class5+Class6+Class7+Class8+Class9, :]
X10 = X[Class1 + Class2 + Class3+Class4+Class5+Class6+Class7+Class8+Class9:Class1 + Class2 + Class3+Class4+Class5+Class6+Class7+Class8+Class9+Class10, :]
acc = []
for i in range(0, 20):
    # 随机抽取样本训练集和测试集样本
    idx = np.random.choice(np.arange(Class1), size = train1, replace = False)
    train_index1 = np.array(idx)
    test_index1 = np.delete(np.arange(Class1), train_index1)

    idx1 = np.random.choice(np.arange(Class2), size = train2, replace = False)
    train_index2 = np.array(idx1)
    test_index2 = np.delete(np.arange(Class2), train_index2)

    idx2 = np.random.choice(np.arange(Class3), size = train3, replace = False)
    train_index3 = np.array(idx2)
    test_index3 = np.delete(np.arange(Class3), train_index3)

    idx3 = np.random.choice(np.arange(Class4), size = train4, replace = False)
    train_index4 = np.array(idx3)
    test_index4 = np.delete(np.arange(Class4), train_index4)

    idx4 = np.random.choice(np.arange(Class5), size = train5, replace = False)
    train_index5 = np.array(idx4)
    test_index5 = np.delete(np.arange(Class5), train_index5)

    idx5 = np.random.choice(np.arange(Class6), size = train6, replace = False)
    train_index6 = np.array(idx5)
    test_index6 = np.delete(np.arange(Class6), train_index6)

    idx6 = np.random.choice(np.arange(Class7), size = train7, replace = False)
    train_index7 = np.array(idx6)
    test_index7 = np.delete(np.arange(Class7), train_index7)

    idx7 = np.random.choice(np.arange(Class8), size = train8, replace = False)
    train_index8 = np.array(idx7)
    test_index8 = np.delete(np.arange(Class8), train_index8)

    idx8 = np.random.choice(np.arange(Class9), size = train9, replace = False)
    train_index9 = np.array(idx8)
    test_index9 = np.delete(np.arange(Class9), train_index9)

    idx9 = np.random.choice(np.arange(Class10), size = train10, replace = False)
    train_index10 = np.array(idx9)
    test_index10 = np.delete(np.arange(Class10), train_index10)

    Data1 = X1[train_index1, :]
    Data2 = X2[train_index2, :]
    Data3 = X3[train_index3, :]
    Data4 = X4[train_index4, :]
    Data5 = X5[train_index5, :]
    Data6 = X6[train_index6, :]
    Data7 = X7[train_index7, :]
    Data8 = X8[train_index8, :]
    Data9 = X9[train_index9, :]
    Data10 = X10[train_index10, :]

    trainingSet = np.vstack((Data1, Data2, Data3, Data4,Data5,Data6,Data7,Data8,Data9,Data10))
    trainingSetX = np.delete(trainingSet, m, axis = 1)
    print(trainingSetX)
    trainingSetY = trainingSet[:, m]

    '''
    Data1 = np.delete(Data1, 9, axis = 1)
    Data2 = np.delete(Data2, 9, axis = 1)
    Data3= np.delete(Data3, 9, axis = 1)
    Data4 = np.delete(Data4, 9, axis = 1)
    Data5 = np.delete(Data5, 9, axis = 1)
    Data6 = np.delete(Data6, 9, axis = 1)
    '''

    testSet1 = np.delete(X1[test_index1, :], m, axis = 1)
    testSet2 = np.delete(X2[test_index2, :], m, axis = 1)
    testSet3 = np.delete(X3[test_index3, :], m, axis = 1)
    testSet4 = np.delete(X4[test_index4, :], m, axis = 1)
    testSet5 = np.delete(X5[test_index5, :], m, axis = 1)
    testSet6 = np.delete(X6[test_index6, :], m, axis = 1)
    testSet7 = np.delete(X7[test_index7, :], m, axis = 1)
    testSet8 = np.delete(X8[test_index8, :], m, axis = 1)
    testSet9 = np.delete(X9[test_index9, :], m, axis = 1)
    testSet10 = np.delete(X10[test_index10, :], m, axis = 1)

    # testSet = np.vstack((testSet1, testSet2))
    # print(testSet)
    # 统计正确数量和计算准确率
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainingSetX, trainingSetY)
    C1 = clf.predict(testSet1)
    add = sum(C1 == 1)
    print(add)
    C2 = clf.predict(testSet2)
    add1 = sum(C2 == 2)
    print(add1)
    C3 = clf.predict(testSet3)
    add2 = sum(C3 == 3)
    print(add2)
    C4 = clf.predict(testSet4)
    add3 = sum(C4 == 4)
    print(add3)
    C5 = clf.predict(testSet5)
    add4 = sum(C5 == 5)
    print(add4)
    C6 = clf.predict(testSet6)
    add5 = sum(C6 == 6)
    print(add5)
    C7 = clf.predict(testSet7)
    add6 = sum(C7 == 7)
    print(add6)
    C8 = clf.predict(testSet8)
    add7 = sum(C8 == 8)
    print(add7)
    C9 = clf.predict(testSet9)
    add8 = sum(C9 == 9)
    print(add8)
    C10 = clf.predict(testSet10)
    add9 = sum(C10 == 10)
    print(add9)
    acc.append((add + add1 + add2 + add3+add4+add5 + add6 + add7 + add8+ add9) / (test1 + test2 + test3 + test4+test5 + test6 + test7 + test8+test9 + test10))
    print("accuracy:{:.2%}".format((add + add1 + add2 + add3+add4+add5 + add6 + add7 + add8+ add9) / (test1 + test2 + test3 + test4+test5 + test6 + test7 + test8+test9 + test10)))
arr_mean = np.mean(acc)
# 求方差
arr_var = np.var(acc)
arr_std = np.std(acc, ddof = 1)
print(arr_mean, arr_var, arr_std)
'''
# 求各类对应属性的均值和方差
Mean1 = np.mean(Data1, axis = 0)
Mean2 = np.mean(Data2, axis = 0)
var1 = np.var(Data1, axis = 0)
var2 = np.var(Data2, axis = 0)
'''
