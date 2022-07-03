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
X = np.loadtxt('3satimage.txt',delimiter=',', dtype=np.int)
# 其中有97
m = X.shape[1] - 1  # 属性数量
print(m)
n = X.shape[0]  # 样本数目
print(n)
Y = X[:, m]
print(Y)
array0 = np.zeros(shape = (0, m+1))
array1 = np.zeros(shape = (0, m+1))
array2 = np.zeros(shape = (0, m+1))
array3 = np.zeros(shape = (0, m+1))
array4 = np.zeros(shape = (0, m+1))
array5 = np.zeros(shape = (0, m+1))
array6 = np.zeros(shape = (0, m+1))
array7 = np.zeros(shape = (0, m+1))
array8 = np.zeros(shape = (0, m+1))
array9 = np.zeros(shape = (0, m+1))
array10 = np.zeros(shape = (0, m+1))
Class1=0
Class2=0
Class3=0
Class4=0
Class5=0
Class6=0
Class7=0
Class8=0
Class9=0
Class10=0
Class11=0

for i in X:
    if i[m]==1:
        Class1=Class1+1
        array0=np.vstack((array0,i))
    elif i[m]==2:
        Class2=Class2+1
        array1=np.vstack((array1,i))
    elif i[m]==3:
        Class3=Class3+1
        array2=np.vstack((array2,i))
    elif i[m]==4:
        Class4=Class4+1
        array3=np.vstack((array3,i))
    elif i[m]==5:
        Class5=Class5+1
        array4=np.vstack((array4,i))
    elif i[m]==7:
        Class6=Class6+1
        array5=np.vstack((array5,i))
print(Class1)
print(Class2)
print(Class3)
print(Class4)
print(Class5)
print(Class6)
X=np.vstack((array0,array1,array2,array3,array4,array5))


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

    Data1 = X1[train_index1, :]
    Data2 = X2[train_index2, :]
    Data3 = X3[train_index3, :]
    Data4 = X4[train_index4, :]
    Data5 = X5[train_index5, :]
    Data6 = X6[train_index6, :]


    trainingSet = np.vstack((Data1, Data2, Data3, Data4,Data5,Data6))
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
    trainSet1=np.delete(Data1, m, axis = 1)
    trainSet2=np.delete(Data2, m, axis = 1)
    trainSet3=np.delete(Data3, m, axis = 1)
    trainSet4=np.delete(Data4, m, axis = 1)
    trainSet5=np.delete(Data5, m, axis = 1)
    trainSet6=np.delete(Data6, m, axis = 1)

    # testSet = np.vstack((testSet1, testSet2))
    # print(testSet)
    # 统计正确数量和计算准确率
    clf = MultinomialNB()
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
    add5 = sum(C6 == 7)
    print(add5)

    acc.append((add + add1 + add2 + add3+add4+add5 ) / (test1+test2+test3+test4+test5+test6))
    print("accuracy:{:.2%}".format((add + add1 + add2 + add3+add4+add5 ) / (test1+test2+test3+test4+test5+test6)))
arr_mean = np.mean(acc)
# 求方差
arr_var = np.var(acc)
arr_std = np.std(acc, ddof = 1)
print(arr_mean, arr_var, arr_std)