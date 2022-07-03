import numpy as np
import math
from sklearn.naive_bayes import GaussianNB


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro


def getRandom(num):
    Ran = np.random.dirichlet(np.ones(num), size = 1)
    Ran = Ran.flatten()
    return Ran


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

X = np.loadtxt('[017]magic(0-1).txt')
m = 10  # 属性数量
n = 1902  # 样本数目
K = 2  # 类标记数量
# 主要过程：分组
T = 3  # 分组数量
Class1 = 1219  #
Class2 = 683  # 342  137  204
# 随机产生多少个和为1的随机数W
G1 = [1, 2, 4, 9]
G2 = [5, 7]
G3 = [0, 3, 6, 8]

# 求类1的分组情况
NewArray = np.ones((Class1, T + 1))
# W = getRandom(m * 2) * 10
W = [0.05206668976087973,0.01099980042259904,0.025547308074078394,0.04638630065998615,0.09490324894684533,0.051454003366956595,0.06355645007060073,0.01936285264777584,0.026411337661442248,0.001459589320526675,0.06719600383195763,0.02414651472015427,0.004015085005398574,0.17301665346170156,0.04370696490667136,0.0010875540628723496,0.14776649278497858,0.05151440261558953,0.055339706002547645,0.0400630416764378]
# 第0组
print(W)
W1 = W[0:4]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 4):
        add1 += W1[j] * X[i, G1[j]]
    NewArray[i][0] = add1
# 第1组
W2 = W[4:6]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 2):
        add2 += W2[j] * X[i, G2[j]]
    NewArray[i][1] = add2
# 第2组
W3 = W[6:10]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 4):
        add3 += W3[j] * X[i, G3[j]]
    NewArray[i][2] = add3

# print(NewArray)

# 求类2的分组情况
NewArray1 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[10:14]
for i in range(Class1, n):
    add1 = 0
    for j in range(0, 4):
        add1 += W4[j] * X[i, G1[j]]
    NewArray1[i - Class1][0] = add1
# 第1组
W5 = W[14:16]
for i in range(Class1, n):
    add2 = 0
    for j in range(0, 2):
        add2 += W5[j] * X[i, G2[j]]
    NewArray1[i - Class1][1] = add2
# 第2组
W6 = W[16:20]
for i in range(Class1, n):
    add3 = 0
    for j in range(0, 4):
        add3 += W6[j] * X[i, G3[j]]
    NewArray1[i - Class1][2] = add3

# print(NewArray1)

# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray, NewArray1))
print(NewArray.shape)
print(NewArray)

Y = NewArray[:, T]

# 去掉类标记
NewArray = np.delete(NewArray, T, axis = 1)
# 取训练集和测试集7:3比例
trainSet1 = NewArray[0:853, :]
trainSet2 = NewArray[1219:1697, :]

testSet1 = NewArray[853:1219, :]
testSet2 = NewArray[1697:1902, :]
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)


clf = GaussianNB()
# 本次代码主要内容是这个，求P(Ai|C)

clf.fit(NewArray, Y)
C1 = clf.predict(trainSet1)
add = sum(C1 == 1)
print(add)
C2 = clf.predict(trainSet2)
add1 = sum(C2 == 2)
print(add1)
print("accuracy:{:.2%}".format((add + add1) / 1331))
