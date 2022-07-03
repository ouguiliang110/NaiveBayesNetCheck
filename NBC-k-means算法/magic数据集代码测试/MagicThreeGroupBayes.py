import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
T= 3  #分组数量
Class1=1219  #
Class2=683   #342  137  204
# 随机产生多少个和为1的随机数W
G1=[1, 2, 4, 9]
G2=[5, 7]
G3=[0, 3, 6, 8]

#求类1的分组情况
NewArray = np.ones((Class1, T+1))
#W = getRandom(m * 2) * 10
W=[7.555825308303979,2.1283340031755045,1.4781833183829558,0.3892312463800208,1.5346806940523776,3.617702589168904,1.7415337095018288,7.300537541975308,1.7290255825203826,9.898503690589198,3.1698664703735826,6.926361951366804,0.7316076907156247,5.522829741472005,5.210100019791337,8.273352382454002,25.404362942755874,1.969993186064575,0.8276704747745849,4.5902974561811565]
# 第0组
print(W)
W1 = W[0:4]
for i in range(0, Class1):
    add1 = 0
    for j in range(0,4):
        add1+=W1[j] * X[i, G1[j]]
    NewArray[i][0]=add1
# 第1组
W2 = W[4:6]
for i in range(0, Class1):
    add2 = 0
    for j in range(0,2):
        add2+=W2[j] * X[i, G2[j]]
    NewArray[i][1]=add2
# 第2组
W3 = W[6:10]
for i in range(0, Class1):
    add3 = 0
    for j in range(0,4):
        add3+=W3[j] * X[i, G3[j]]
    NewArray[i][2]=add3

#print(NewArray)

#求类2的分组情况
NewArray1 = np.ones((Class2, T+1))*2
# 第0组
W4 = W[10:14]
for i in range(Class1, n):
    add1 = 0
    for j in range(0,4):
        add1+=W4[j] * X[i, G1[j]]
    NewArray1[i-Class1][0]=add1
# 第1组
W5 = W[14:16]
for i in range(Class1, n):
    add2 = 0
    for j in range(0,2):
        add2+=W5[j] * X[i, G2[j]]
    NewArray1[i-Class1][1]=add2
# 第2组
W6 = W[16:20]
for i in range(Class1, n):
    add3 = 0
    for j in range(0,4):
        add3+=W6[j] * X[i, G3[j]]
    NewArray1[i-Class1][2]=add3

#print(NewArray1)

#合并两个数组，得到真正的合并数据结果
NewArray=np.vstack((NewArray,NewArray1))
print(NewArray.shape)
print(NewArray)

NewArray = np.delete(NewArray, T, axis = 1)
df=pd.DataFrame(NewArray)
sns.heatmap(df.corr(), annot = True)
plt.show()


# 去掉类标记
#NewArray = np.delete(NewArray, T, axis = 1)
# 取训练集和测试集7:3比例
trainSet1 = NewArray[0:853, :]
trainSet2 = NewArray[1219:1697, :]


testSet1 = NewArray[853:1219, :]
testSet2 = NewArray[1697:1902, :]
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
print(Mean1)
Mean2 = np.mean(trainSet2, axis = 0)
var1 = np.var(trainSet1, axis = 0)
var2 = np.var(trainSet2, axis = 0)

# 先求P(C)
Pro1 = (853 + 1) / (1331 + 2)
Pro2 = (478 + 1) / (1331 + 2)
# print(Pro1)
# print(Pro2)

# 本次代码主要内容是这个，求P(Ai|C)

# 统计正确数量和计算准确率
add = 0
for i in range(0, 853):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet1[i][j], Mean2[j], var2[j])
    if Pro1*sum >= Pro2*sum1:
        add += 1
    elif Pro1*sum < Pro2*sum1:
        add += 0
print("第一类正确数量(总数365)：")
print(add)
add1=0
for i in range(0, 478):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet2[i][j], Mean2[j], var2[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet2[i][j], Mean1[j], var1[j])
    if Pro2*sum >= Pro1*sum1:
        add1 += 1
    elif Pro2*sum < Pro1*sum1:
        add1 += 0
print("第二类正确数量(总数205)：")
print(add1)

#准确率
print("accuracy:{:.2%}".format((add+add1)/1331))

