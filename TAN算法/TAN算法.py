import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
import seaborn as sns
import pandas as pd


X = np.loadtxt('../TAN算法/数据集/[010]glass(0-1).txt')
m = X.shape[1] - 1  # 属性数量
print(m)
n = X.shape[0]  # 样本数目 其中第一类1219，第二类683
vector_data = X[:, :-1]
# 提取label类别
label_data = X[:, -1]
print(label_data)
# 用两个类来完成
'''
p=0
array = np.zeros(shape = (0, m))
for i in vector_data:
    k = 10
    d1 = pd.cut(i, k, labels = range(k))
    #d1=np.append(d1,label_data[p])
    array = np.vstack((array, d1))
# 以《统计学习方法》中的例4.1计算，为方便计算，将例子中"S"设为0，“M"设为1。
# 提取特征向量
vector_data=array[:,:-1]
print(vector_data)
bayes = NBayes(smooth = 1)
bayes.train(vector_data,label_data)
p=0
acc=0
for i in vector_data:
    if bayes.predict(i)[0]==label_data[p]:
        acc+=1
    print(bayes.predict(i)[0],label_data[p])
    p+=1
print(acc/n)
'''

'''
Class1 = 0
Class2 = 0
    for i in array:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

train1 = int(Class1 * 0.7)
test1 = Class1 - train1

train2 = int(Class2 * 0.7)
test2 = Class2 - train2

X1=vector_data[0:Class1,:]
X2=vector_data[Class1:Class1+Class2,:]

testSet1=X1[train1:Class1,:]
testSet2=X2[train2:Class2,:]
p=0
correct1=0
correct2=0
for i in testSet1:
    if bayes.predict(i)[0]==1:
        correct1+=1
print(correct1)
for i in testSet2:
    if bayes.predict(i)[0]==2:
        correct2+=1
print(correct2)
print((correct1+correct2)/(test1+test2))
'''
# data = pd.DataFrame(vector_data)
array1 = np.zeros(shape = (0, n))
for n in range(0, m):
    k = 10
    d1 = pd.cut(vector_data[:, n], k, labels = range(k))
    array1 = np.vstack((array1, d1))
array1 = np.vstack((array1, label_data))
X = array1.T
print(X)


# 采用贝叶斯估计计算条件概率和先验概率，此时拉普拉斯平滑参数为1，为0时即为最大似然估

SetClass = set(X[:, m])
SetClass = list(map(int, SetClass))
print(SetClass)
K = len(SetClass)  # 类标记数量

newarray=[np.zeros(shape=[0,m+1])]*20
for i in X:
    for j in SetClass:
        if i[m] == j:
            newarray[j] = np.vstack((newarray[j], i))

NewArray=np.zeros(shape=[0,m+1])
for i in SetClass:
    NewArray=np.vstack((NewArray,newarray[i]))
print(NewArray)
print(NewArray.shape)

X=NewArray

NumClass = [0] * K
# 初始化U
p = 0
for i in X:
    for j in range(0, K):
        if i[m] == SetClass[j]:
            NumClass[j] = NumClass[j] + 1
    p = p + 1
print(NumClass)

arr_train=[]
arr_test=[]

for k in range(0,20):
    train = []
    trainNum = 0
    val = []
    valNum = 0
    test = []
    testNum = 0
    for i in range(0, K):
        train.append(int(NumClass[i] * 0.5))
        trainNum += int(NumClass[i] * 0.5)

        val.append(int(NumClass[i] * 0.2))
        valNum += int(NumClass[i] * 0.2)

        test.append(NumClass[i] - train[i] - val[i])
        testNum += NumClass[i] - train[i] - val[i]

    train_index = []
    val_index = []
    test_index = []
    for i in range(0, K):
        idx = np.random.choice(np.arange(NumClass[i]), size = train[i], replace = False)
        train_index.append(np.array(idx))
        val_index.append(
            np.random.choice(np.delete(np.arange(NumClass[i]), train_index[i]), size = val[i], replace = False))
        test_index.append(np.delete(np.arange(NumClass[i]), np.append(train_index[i], val_index[i])))

    dividX = []
    p2 = 0
    for i in range(0, K):
        dividX.append(X[p2:p2 + NumClass[i], :])
        p2 = p2 + NumClass[i]

    trainSet = []
    for i in range(0, K):
        trainSet.append(dividX[i][train_index[i], :])
    TrainSet = np.zeros((0, m + 1))
    for i in range(0, K):
        TrainSet = np.vstack((TrainSet, trainSet[i]))
    Y = TrainSet[:, m]
    TrainSet = np.delete(TrainSet, m, axis = 1)
    for i in range(0, K):
        trainSet[i] = np.delete(trainSet[i], m, axis = 1)

    testSet = []
    for i in range(0, K):
        testSet.append(np.delete(dividX[i][test_index[i], :], m, axis = 1))
    valSet = []
    for i in range(0, K):
        valSet.append(np.delete(dividX[i][val_index[i], :], m, axis = 1))
    clf = ComplementNB()

    clf.fit(TrainSet, Y)

    correct = 0
    for i in range(0, K):
        C = clf.predict(testSet[i])
        print(C)
        print(SetClass[i])
        correct += sum(C == SetClass[i])
        print(sum(C == SetClass[i]))
        # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
    testacc = correct / testNum
    arr_test.append(testacc)
    print("test accuracy:{:.2%}".format(testacc))
    print("---------------------------")
    correct1 = 0
    for i in range(0, K):
        C = clf.predict(trainSet[i])
        print(C)
        print(SetClass[i])
        correct1 += sum(C == SetClass[i])
        print(sum(C == SetClass[i]))
        # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
    trainacc = correct1 / trainNum
    arr_train.append(trainacc)
    print("train accuracy:{:.2%}".format(trainacc))

print(arr_train)
print(arr_test)

# 求标准差
arr_mean = np.mean(arr_train)
arr_std = np.std(arr_train, ddof = 1)
arr_mean1=np.mean(arr_test)
arr_std1=np.std(arr_test,ddof = 1)


print("训练集平均标准差",arr_mean,arr_std)
print("测试集平均标准差",arr_mean1,arr_std1)
