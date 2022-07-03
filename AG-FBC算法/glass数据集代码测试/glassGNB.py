import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
import random


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    a=1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro=1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro
def getRandom(num):
    Ran = np.random.dirichlet(np.ones(num)*2, size = 1)
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
X=np.loadtxt('[010]glass(0-1).txt')
# 其中有97
T=3  # 组数量大小
m=9  # 属性数量
n=214  # 样本数目
K=6  # 类标记数量
Y=X[:,9]
X=np.delete(X, 9, axis = 1)

# 取训练集和测试集7：3比例，70 76 17 29 13 9  训练集149，测试65
trainSet1=X[0:49, :]  # 49
trainSet2=X[70:123, :]  # 53
trainSet3=X[146:158, :]  # 12
trainSet4=X[163:183, :]  # 20
trainSet5=X[192:201, :]  # 9
trainSet6=X[205:211, :]  # 6
# trainingSet = np.vstack((Data1, Data2))
# print(trainingSet)
testSet1= X[49:70, :]  # 21
testSet2= X[123:146, :]  # 23
testSet3= X[158:163, :]  # 5
testSet4= X[183:192, :]  # 9
testSet5= X[201:205, :]  # 4
testSet6= X[211:214, :]  # 3
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

# 统计正确数量和计算准确率
# 计算第一类

clf = GaussianNB()
clf.fit(X,Y)
C1=clf.predict(trainSet1)
add=sum(C1==1)
print(add)
C2=clf.predict(trainSet2)
add1=sum(C2==2)
print(add1)
C1=clf.predict(trainSet3)
add2=sum(C1==3)
print(add2)
C2=clf.predict(trainSet4)
add3=sum(C2==4)
print(add3)
C1=clf.predict(trainSet5)
add4=sum(C1==5)
print(add4)
C2=clf.predict(trainSet6)
add5=sum(C2==6)
print(add5)
print("accuracy:{:.2%}".format((add+add1+add2+add3+add4+add5)/149))
