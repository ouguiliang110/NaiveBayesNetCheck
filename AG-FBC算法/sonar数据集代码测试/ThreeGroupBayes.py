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

X = np.loadtxt('[023]sonar(0-1).txt')
# 其中有97
m = 60  # 属性数量
n = 208  # 样本数目
K = 2  # 类标记数量
# 主要过程：分组
Class1=97
Class2=111
# 随机产生多少个和为1的随机数W
G1=[15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 33, 34, 35, 36]
G2=[5, 14, 23, 26, 27, 28, 29, 30, 31, 37, 38, 39]
G3=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

#求类1的分组情况
NewArray = np.ones((97, 4))
# 第0组
W1 = getRandom(14)
for i in range(0, 97):
    add1 = 0
    for j in range(0,14):
        add1+=W1[j] * X[i, G1[j]]
    NewArray[i][0]=add1
# 第1组
W2 = getRandom(12)
for i in range(0, 97):
    add2 = 0
    for j in range(0,12):
        add2+=W2[j] * X[i, G2[j]]
    NewArray[i][1]=add2
# 第2组
W3 = getRandom(32)
for i in range(0, 97):
    add3 = 0
    for j in range(0,32):
        add3+=W3[j] * X[i, G3[j]]
    NewArray[i][2]=add3

#print(NewArray)

#求类2的分组情况
NewArray1 = np.ones((111, 4))*2
# 第0组
#W1 = getRandom(14)
for i in range(Class1, n):
    add1 = 0
    for j in range(0,14):
        add1+=W1[j] * X[i, G1[j]]
    NewArray1[i-Class1][0]=add1
# 第1组
#W2 = getRandom(12)
for i in range(Class1, n):
    add2 = 0
    for j in range(0,12):
        add2+=W2[j] * X[i, G2[j]]
    NewArray1[i-Class1][1]=add2
# 第2组
#W3 = getRandom(32)
for i in range(Class1, n):
    add3 = 0
    for j in range(0,32):
        add3+=W3[j] * X[i, G3[j]]
    NewArray1[i-Class1][2]=add3

#print(NewArray1)

#合并两个数组，得到真正的合并数据结果
NewArray=np.vstack((NewArray,NewArray1))
print(NewArray)


Y=NewArray[:,3]
# 去掉类标记
NewArray = np.delete(NewArray, 3, axis = 1)

# 取训练集和测试机7：3比例
Data1 = NewArray[0:70, :]
Data2 = NewArray[97:174, :]
trainingSet = np.vstack((Data1, Data2))
#print(trainingSet)
testSet1 = NewArray[70:97, :]
testSet2 = NewArray[174:208, :]
testSet = np.vstack((testSet1, testSet2))
#print(testSet)



# 本次代码主要内容是这个，求P(Ai|C)
clf=GaussianNB()
clf.fit(NewArray,Y)
C1=clf.predict(testSet1)
add=sum(C1==1)
print(add)
C2=clf.predict(testSet2)
add1=sum(C2==2)
print(add1)
print("accuracy:{:.2%}".format((add+add1)/61))

