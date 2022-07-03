import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

X = np.loadtxt('[010]glass(0-1).txt')
Y = X[:, 9]
X = np.delete(X, 9, axis = 1)
# 其中有97
T = 3  # 组数量大小
m = 9  # 属性数量
n = 214  # 样本数目
K = 6  # 类标记数量

Class1 = 70
Class2 = 76
Class3 = 17
Class4 = 29
Class5 = 13
Class6 = 9
data = pd.DataFrame(X)
array = np.zeros(shape = (0, 9))
for i in X:
    k = 15
    d1 = pd.cut(i, k, labels = range(k))
    array = np.vstack((array, d1))
#print(array)

X1 = array[0:70, :]
X2 = array[70:146, :]
X3 = array[146:163, :]
X4 = array[163:192, :]
X5 = array[192:205, :]
X6 = array[205:214, :]
#print(X6)

# 随机抽取样本训练集和测试集样本
for i in range(0, 10):
    idx1 = np.random.choice(np.arange(70), size = 49, replace = False)
    train_index1 = np.array(idx1)
    test_index1 = np.delete(np.arange(70), train_index1)

    idx2 = np.random.choice(np.arange(76), size =53, replace = False)
    train_index2 = np.array(idx2)
    test_index2 = np.delete(np.arange(76), train_index2)

    idx3 = np.random.choice(np.arange(17), size = 12, replace = False)
    train_index3 = np.array(idx3)
    test_index3 = np.delete(np.arange(17), train_index3)

    idx4 = np.random.choice(np.arange(29), size = 20, replace = False)
    train_index4 = np.array(idx4)
    test_index4 = np.delete(np.arange(29), train_index4)

    idx5 = np.random.choice(np.arange(13), size = 9, replace = False)
    train_index5 = np.array(idx5)
    test_index5 = np.delete(np.arange(13), train_index5)


    idx6 = np.random.choice(np.arange(9), size = 6, replace = False)
    train_index6 = np.array(idx6)
    test_index6 = np.delete(np.arange(9), train_index6)
    #print(idx6)
    # 取训练集和测试集7：3比例
    trainSet1 = X1[train_index1, :]
    trainSet2 = X2[train_index2, :]
    trainSet3 = X3[train_index3, :]
    trainSet4 = X4[train_index4, :]
    trainSet5 = X5[train_index5, :]
    #print(trainSet5)
    trainSet6 = X6[train_index6, :]

    #trainingSet = np.vstack((Data1, Data2))
    # print(trainingSet)
    testSet1 = X1[test_index1, :]
    testSet2 = X2[test_index2, :]
    testSet3 = X3[test_index3, :]
    testSet4 = X4[test_index4, :]
    testSet5 = X5[test_index5, :]
    testSet6 = X6[test_index6, :]

    # testSet = np.vstack((testSet1, testSet2))
    # print(testSet)
    # 统计正确数量和计算准确率
    clf = MultinomialNB()
    clf.fit(array, Y)
    C1 = clf.predict(testSet1)
    add = sum(C1 == 1)
    #print(add)
    C2 = clf.predict(testSet2)
    add1 = sum(C2 == 2)
    #print(add1)
    C1 = clf.predict(testSet3)
    add2 = sum(C1 == 3)
   # print(add2)
    C2 = clf.predict(testSet4)
    add3 = sum(C2 == 4)
    #print(add3)
    C1 = clf.predict(testSet5)
    add4 = sum(C1 == 5)
   # print(add4)
    C2 = clf.predict(testSet6)
    add5 = sum(C2 == 6)
    #print(add5)
    print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5) / 65))
