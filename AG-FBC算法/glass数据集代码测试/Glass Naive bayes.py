import numpy as np
import math
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

# 求各类对应属性的均值和方差
Mean1=np.mean(trainSet1, axis = 0)
Mean2=np.mean(trainSet2, axis = 0)
Mean3=np.mean(trainSet3, axis = 0)
Mean4=np.mean(trainSet4, axis = 0)
Mean5=np.mean(trainSet5, axis = 0)
Mean6=np.mean(trainSet6, axis = 0)

var1=np.var(trainSet1, axis = 0)
var2=np.var(trainSet2, axis = 0)
var3=np.var(trainSet3, axis = 0)
var4=np.var(trainSet4, axis = 0)
var5=np.var(trainSet5, axis = 0)
var6=np.var(trainSet6, axis = 0)
print(var1, var2, var3, var4, var5, var6)

# 先求P(C)
Pro1=(49 + 1) / (149 + 6)
Pro2=(53 + 1) / (149 + 6)
Pro3=(12 + 1) / (149 + 6)
Pro4=(20 + 1) / (149 + 6)
Pro5=(9 + 1) / (149 + 6)
Pro6=(6 + 1) / (149 + 6)
# print(Pro1)
# print(Pro2)

# 本次代码主要内容是这个，求P(Ai|C)

# 统计正确数量和计算准确率
# 计算第一类
add=0
for i in range(0, 21):
    sum=1
    for j in range(0, m):
        sum*=getPro(trainSet1[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, m):
        sum1*=getPro(trainSet1[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, m):
        sum2*=getPro(trainSet1[i][j], Mean3[j], var3[j])
    sum3=  1
    for j in range(0, m):
        sum3*=getPro(trainSet1[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, m):
        sum4*=getPro(trainSet1[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, m):
        sum5*=getPro(trainSet1[i][j], Mean6[j], var6[j])
    if (Pro1 * sum >= Pro2 * sum1) & (Pro1 * sum >= Pro3 * sum2) & (Pro1 * sum >= Pro4 * sum3) & (Pro1 * sum >= Pro5 * sum4) & (Pro1 * sum >= Pro6 * sum5):
        add+=1
    else:
        add+=0
print("第一类正确数量(总数21)：")
print(add)

# 计算第二类
add1=0
for i in range(0, 23):
    sum=1
    for j in range(0, m):
        sum*=getPro(trainSet2[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, m):
        sum1*=getPro(trainSet2[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, m):
        sum2*=getPro(trainSet2[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, m):
        sum3*=getPro(trainSet2[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, m):
        sum4*=getPro(trainSet2[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, m):
        sum5*=getPro(trainSet2[i][j], Mean6[j], var6[j])
    if (Pro2 * sum1 >= Pro1 * sum) & (Pro2 * sum1 >= Pro3 * sum2) & (Pro2 * sum1 >= Pro4 * sum3) & (Pro2 * sum1 >= Pro5 * sum4) & (Pro2 * sum1 >= Pro6 * sum5):
        add1+=1
    else:
        add1+=0
print("第二类正确数量(总数23)：")
print(add1)

# 计算第三类
add2=0
for i in range(0, 5):
    sum=1
    for j in range(0, m):
        sum*=getPro(trainSet3[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, m):
        sum1*=getPro(trainSet3[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, m):
        sum2*=getPro(trainSet3[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, m):
        sum3*=getPro(trainSet3[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, m):
        sum4*=getPro(trainSet3[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, m):
        sum5*=getPro(trainSet3[i][j], Mean6[j], var6[j])
    if (Pro3 * sum2 >= Pro1 * sum) & (Pro3 * sum2 >= Pro2 * sum1) & (Pro3 * sum2 >= Pro4 * sum3) & (Pro3 * sum2 >= Pro5 * sum4) & (Pro3 * sum2 >= Pro6 * sum5):
        add2+=1
    else:
        add2+=0
print("第三类正确数量(总数5)：")
print(add2)

add3=0
for i in range(0, 9):
    sum=1
    for j in range(0, m):
        sum*=getPro(trainSet4[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, m):
        sum1*=getPro(trainSet4[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, m):
        sum2*=getPro(trainSet4[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, m):
        sum3*=getPro(trainSet4[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, m):
        sum4*=getPro(trainSet4[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, m):
        sum5*=getPro(trainSet4[i][j], Mean6[j], var6[j])
    if (Pro4 * sum3 >= Pro1 * sum) & (Pro4 * sum3 >= Pro2 * sum1) & (Pro4 * sum3 >= Pro3 * sum2) & (Pro4 * sum3 >= Pro5 * sum4) & (Pro4 * sum3 >= Pro6 * sum5):
        add3+=1
    else:
        add3+=0
print("第四类正确数量(总数9)：")
print(add3)

add4=0
for i in range(0, 4):
    sum=1
    for j in range(0, m):
        sum*=getPro(trainSet5[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, m):
        sum1*=getPro(trainSet5[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, m):
        sum2*=getPro(trainSet5[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, m):
        sum3*=getPro(trainSet5[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, m):
        sum4*=getPro(trainSet5[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, m):
        sum5*=getPro(trainSet5[i][j], Mean6[j], var6[j])
    if (Pro5 * sum4 >= Pro1 * sum) & (Pro5 * sum4 >= Pro2 * sum1) & (Pro5 * sum4 >= Pro3 * sum2) & (Pro5 * sum4 >= Pro4 * sum3) & (Pro5 * sum4 >= Pro6 * sum5):
        add4+=1
    else:
        add4+=0
print("第五类正确数量(总数4)：")
print(add4)

add5=0
for i in range(0, 3):
    sum=1
    for j in range(0, m):
        sum*=getPro(trainSet6[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, m):
        sum1*=getPro(trainSet6[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, m):
        sum2*=getPro(trainSet6[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, m):
        sum3*=getPro(trainSet6[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, m):
        sum4*=getPro(trainSet6[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, m):
        sum5*=getPro(trainSet6[i][j], Mean6[j], var6[j])
    if (Pro6 * sum5 >= Pro1 * sum) & (Pro6 * sum5 >= Pro2 * sum1) & (Pro6 * sum5 >= Pro3 * sum2) & (Pro6 * sum5 >= Pro4 * sum3) & (Pro6 * sum5 >= Pro5 * sum4):
        add5+=1
    else:
        add5+=0
print("第六类正确数量(总数3)：")
print(add5)

# 准确率
print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5) / 65))
