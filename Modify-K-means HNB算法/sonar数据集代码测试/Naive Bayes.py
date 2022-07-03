import numpy as np
import math
import random


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro


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

X = np.loadtxt('[023]sonar(0-1).txt')
# 其中有97
m = 60  # 属性数量
n = 208  # 样本数目
K = 2  # 类标记数量

# 去掉类标记
X = np.delete(X, 60, axis = 1)

# 取训练集和测试机7：3比例
Data1 = X[0:70, :]
Data2 = X[97:174, :]
trainingSet = np.vstack((Data1, Data2))
# print(trainingSet)
testSet1 = X[70:97, :]
testSet2 = X[174:208, :]
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

# 求各类对应属性的均值和方差
Mean1 = np.mean(Data1, axis = 0)
Mean2 = np.mean(Data2, axis = 0)
var1 = np.var(Data1, axis = 0)
var2 = np.var(Data2, axis = 0)

# 先求P(C)
Pro1 = (70 + 1) / (147 + 1)
Pro2 = (77 + 1) / (147 + 1)
# print(Pro1)
# print(Pro2)

# 本次代码主要内容是这个，求P(Ai|C)

# 统计正确数量和计算准确率
add = 0
for i in range(0, 27):
    sum = 1
    for j in range(0, 60):
        sum *= getPro(testSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, 60):
        sum1 *= getPro(testSet1[i][j], Mean2[j], var2[j])
    if Pro1*sum >= Pro2*sum1:
        add += 1
    elif Pro1*sum < Pro2*sum1:
        add += 0
print("第一类正确数量(总数27)：")
print(add)
add1=0
for i in range(0, 34):
    sum = 1
    for j in range(0, 60):
        sum *= getPro(testSet2[i][j], Mean2[j], var2[j])
    sum1 = 1
    for j in range(0, 60):
        sum1 *= getPro(testSet2[i][j], Mean1[j], var1[j])
    if Pro2*sum >= Pro1*sum1:
        add1 += 1
    elif Pro2*sum < Pro1*sum1:
        add1 += 0
print("第二类正确数量(总数34)：")
print(add1)
#准确率
print("accuracy:{:.2%}".format((add+add1)/61))

