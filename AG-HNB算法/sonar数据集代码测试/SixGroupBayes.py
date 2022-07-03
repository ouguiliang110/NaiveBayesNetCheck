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

X = np.loadtxt('[023]sonar(0-1).txt')
# 其中有97
m = 60  # 属性数量
n = 208  # 样本数目
K = 2  # 类标记数量
# 主要过程：分组

# 随机产生多少个和为1的随机数W
G1=[16,17,19]
G2=[15,18,20,21,24,25,33,34,35,36]
G3=[11,14,22,23,26,27,28,29,30,31,37,38,39]
G4=[0,1,2,3,7,9,12,32,40,41,43,45,46,47,50,51,54,57,58,59]
G5=[44,55]
G6=[4,5,6,8,10,13,42,48,49,52,53,56]
#求类1的分组情况
NewArray = np.ones((97, 7))
# 第0组
W1 = getRandom(3)
for i in range(0, 97):
    add1 = 0
    for j in range(0,3):
        add1+=W1[j] * X[i, G1[j]]
    NewArray[i][0]=add1
# 第1组
W2 = getRandom(10)
for i in range(0, 97):
    add2 = 0
    for j in range(0,10):
        add2+=W2[j] * X[i, G2[j]]
    NewArray[i][1]=add2
# 第2组
W3 = getRandom(13)
for i in range(0, 97):
    add3 = 0
    for j in range(0,13):
        add3+=W3[j] * X[i, G3[j]]
    NewArray[i][2]=add3
# 第3组
W4 = getRandom(20)
for i in range(0, 97):
    add4 = 0
    for j in range(0,20):
        add4+=W4[j] * X[i, G4[j]]
    NewArray[i][3]=add4
# 第4组
W5 = getRandom(2)
for i in range(0, 97):
    add5 = 0
    for j in range(0,2):
        add5+=W5[j] * X[i, G5[j]]
    NewArray[i][4]=add5
# 第5组
W6 = getRandom(12)
for i in range(0, 97):
    add6 = 0
    for j in range(0,12):
        add6+=W6[j] * X[i, G6[j]]
    NewArray[i][5]=add6
#print(NewArray)

#求类2的分组情况
NewArray1 = np.ones((111, 7))*2
# 第0组
W1 = getRandom(3)
for i in range(0, 111):
    add1 = 0
    for j in range(0,3):
        add1+=W1[j] * X[i, G1[j]]
    NewArray1[i][0]=add1
# 第1组
W2 = getRandom(10)
for i in range(0, 111):
    add2 = 0
    for j in range(0,10):
        add2+=W2[j] * X[i, G2[j]]
    NewArray1[i][1]=add2
# 第2组
W3 = getRandom(13)
for i in range(0, 111):
    add3 = 0
    for j in range(0,13):
        add3+=W3[j] * X[i, G3[j]]
    NewArray1[i][2]=add3
# 第3组
W4 = getRandom(20)
for i in range(0, 111):
    add4 = 0
    for j in range(0,20):
        add4+=W4[j] * X[i, G4[j]]
    NewArray1[i][3]=add4
# 第4组
W5 = getRandom(2)
for i in range(0, 111):
    add5 = 0
    for j in range(0,2):
        add5+=W5[j] * X[i, G5[j]]
    NewArray1[i][4]=add5
# 第5组
W6 = getRandom(12)
for i in range(0, 111):
    add6 = 0
    for j in range(0,12):
        add6+=W6[j] * X[i, G6[j]]
    NewArray1[i][5]=add6
#print(NewArray1)

#合并两个数组，得到真正的合并数据结果
NewArray=np.vstack((NewArray,NewArray1))
print(NewArray)



# 去掉类标记
NewArray = np.delete(NewArray, 6, axis = 1)

# 取训练集和测试机7：3比例
Data1 = NewArray[0:70, :]
Data2 = NewArray[97:174, :]
trainingSet = np.vstack((Data1, Data2))
#print(trainingSet)
testSet1 = NewArray[70:97, :]
testSet2 = NewArray[174:208, :]
testSet = np.vstack((testSet1, testSet2))
#print(testSet)

# 求各类对应属性的均值和方差
Mean1 = np.mean(Data1, axis = 0)
print(Mean1)
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
    for j in range(0, 6):
        sum *= getPro(testSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, 6):
        sum1 *= getPro(testSet1[i][j], Mean2[j], var2[j])
    #print(sum)
    #print(sum1)
    if Pro1*sum > Pro2*sum1:
        add += 1
    elif Pro1*sum < Pro2*sum1:
        add += 0
print("第一类正确数量(总数27)：")
print(add)
add1 = 0
for i in range(0, 34):
    sum = 1
    for j in range(0, 6):
        sum *= getPro(testSet2[i][j], Mean2[j], var2[j])
    sum1 = 1
    for j in range(0, 6):
        sum1 *= getPro(testSet2[i][j], Mean1[j], var1[j])
    if Pro2*sum > Pro1*sum1:
        add1 += 1
    elif Pro2*sum < Pro1*sum1:
        add1 += 0
print("第二类正确数量(总数34)：")
print(add1)
# 准确率
print("accuracy:{:.2%}".format((add + add1) / 61))