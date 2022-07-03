import numpy as np
import math
from sklearn.naive_bayes import GaussianNB

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
clf = GaussianNB()
X=np.loadtxt('[028]wineQR(0-1).txt')
# 其中有97
T = 3  # 组数量大小
m = 11  # 属性数量
n = 1599  # 样本数目
K = 6  # 类标记数量
Y=X[:,m]
X=np.delete(X, 11, axis = 1)

# 取训练集和测试集7：3比例，10,53,681,638,199,18  训练集1118，测试481
trainSet1=X[0:7, :]  # 7
trainSet2=X[10:47, :]  # 37
trainSet3=X[63:540, :]  # 477
trainSet4=X[744:1190, :]  # 446
trainSet5=X[1382:1521, :]  # 139
trainSet6=X[1581:1593, :]  # 12
# trainingSet = np.vstack((Data1, Data2))
# print(trainingSet)
testSet1= X[7:10, :]  # 3
testSet2= X[47:63, :]  # 16
testSet3= X[540:744, :]  # 204
testSet4= X[1190:1382, :]  # 192
testSet5= X[1521:1581, :]  # 60
testSet6= X[1593:1599, :]  # 6
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

clf.fit(X,Y)
C1=clf.predict(testSet1)
add=sum(C1==1)
print(add)
C2=clf.predict(testSet2)
add1=sum(C2==2)
print(add1)
C1=clf.predict(testSet3)
add2=sum(C1==3)
print(add2)
C2=clf.predict(testSet4)
add3=sum(C2==4)
print(add3)
C1=clf.predict(testSet5)
add4=sum(C1==5)
print(add4)
C2=clf.predict(testSet6)
add5=sum(C2==6)
print(add5)
print("accuracy:{:.2%}".format((add+add1+add2+add3+add4+add5)/481))