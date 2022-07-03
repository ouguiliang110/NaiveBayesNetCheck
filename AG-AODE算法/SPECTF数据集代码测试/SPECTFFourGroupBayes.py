import numpy as np
import math


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

X = np.loadtxt('[024]SPECTF(0-1).txt')
# 其中有97
m = 44  # 属性数量
n = 267  # 样本数目
T = 4
K = 2  # 类标记数量
Class1 = 212
Class2 = 55
# 主要过程：分组

# 随机产生多少个和为1的随机数W
G1 = [0, 6, 16, 19, 26]
G2 = [10, 13, 31, 33, 34, 35, 40, 41]
G3 = [1, 2, 4, 5, 7, 8, 15, 17, 18, 21, 22, 23, 36, 37, 38, 39]
G4 = [3, 9, 11, 12, 14, 20, 24, 25, 27, 28, 29, 30, 32, 42, 43]
# 求类1的分组情况
NewArray = np.ones((Class1, T+1))
# 第0组
W1 = getRandom(5)
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 5):
        add1 += W1[j] * X[i, G1[j]]
    NewArray[i][0] = add1
# 第1组
W2 = getRandom(8)
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 8):
        add2 += W2[j] * X[i, G2[j]]
    NewArray[i][1] = add2
# 第2组
W3 = getRandom(16)
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 16):
        add3 += W3[j] * X[i, G3[j]]
    NewArray[i][2] = add3
# 第3组
W4 = getRandom(15)
for i in range(0, Class1):
    add4 = 0
    for j in range(0, 15):
        add4 += W4[j] * X[i, G4[j]]
    NewArray[i][3] = add4
# print(NewArray)

# 求类2的分组情况
NewArray1 = np.ones((Class2, 5)) * 2
# 第0组
W1 = getRandom(5)
for i in range(0, Class2):
    add1 = 0
    for j in range(0, 5):
        add1 += W1[j] * X[i, G1[j]]
    NewArray1[i][0] = add1
# 第1组
W2 = getRandom(8)
for i in range(0, Class2):
    add2 = 0
    for j in range(0, 8):
        add2 += W2[j] * X[i, G2[j]]
    NewArray1[i][1] = add2
# 第2组
W3 = getRandom(16)
for i in range(0, Class2):
    add3 = 0
    for j in range(0, 16):
        add3 += W3[j] * X[i, G3[j]]
    NewArray1[i][2] = add3
# 第3组
W4 = getRandom(15)
for i in range(0, Class2):
    add4 = 0
    for j in range(0, 15):
        add4 += W4[j] * X[i, G4[j]]
    NewArray1[i][3] = add4
# print(NewArray1)

# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray, NewArray1))
print(NewArray)

# 去掉类标记
NewArray = np.delete(NewArray, 4, axis = 1)

# 取训练集和测试集7：3比例
Data1 = NewArray[0:147, :]
Data2 = NewArray[212:247, :]
trainingSet = np.vstack((Data1, Data2))
# print(trainingSet)
testSet1 = NewArray[147:212, :]
testSet2 = NewArray[247:267, :]
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

# 求各类对应属性的均值和方差
Mean1 = np.mean(Data1, axis = 0)
# print(Mean1)
Mean2 = np.mean(Data2, axis = 0)
var1 = np.var(Data1, axis = 0)
var2 = np.var(Data2, axis = 0)

# 先求P(C)
Pro1 = (147 + 1) / (182 + 2)
Pro2 = (35 + 1) / (182 + 2)
# print(Pro1)
# print(Pro2)

# 本次代码主要内容是这个，求P(Ai|C)

# 通过朴素贝叶斯算法得到分类器的准确率
add = 0
for i in range(0, 65):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet1[i][j], Mean2[j], var2[j])
    if Pro1 * sum >= Pro2 * sum1:
        add += 1
    elif Pro1 * sum < Pro2 * sum1:
        add += 0
print("第一类正确数量(总数65)：")
print(add)
add1 = 0
for i in range(0, 20):
    sum = 1
    for j in range(0, T):
        sum *= getPro(testSet2[i][j], Mean2[j], var2[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(testSet2[i][j], Mean1[j], var1[j])
    if Pro2 * sum >= Pro1 * sum1:
        add1 += 1
    elif Pro2 * sum < Pro1 * sum1:
        add1 += 0

print("第二类正确数量(总数20)：")
print(add1)
# 准确率
print("accuracy:{:.2%}".format((add + add1) / 85))
