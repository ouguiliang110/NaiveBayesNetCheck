import numpy as np
import math


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
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

X = np.loadtxt('[024]SPECTF(0-1).txt')
# 其中有97
m = 44  # 属性数量
n = 267  # 样本数目
T = 3
K = 2  # 类标记数量
Class1 = 212
Class2 = 55
# 主要过程：分组

# 随机产生多少个和为1的随机数W
G1 = [6, 8, 10, 11, 16, 25, 31, 33, 34, 35, 38, 40, 41, 42]
G2 = [0, 3, 4, 9, 12, 14, 19, 20, 23, 24, 26, 28, 29, 32, 43]
G3 = [1, 2, 5, 7, 13, 15, 17, 18, 21, 22, 27, 30, 36, 37, 39]
W=getRandom(m*2)*100
print(W)
W=[0.017296475136752767,0.015383772918940171,0.8145406612964592,0.7867452380867964,0.2617431417210009,0.1992643830603857,0.7727684437708198,0.13452328589051404,0.5503852603839754,1.7168936626646216,0.0667637626265985,0.2112712051040477,0.47171338269431357,0.614452073991371,1.4161439041906025,4.0721175173729485,1.81143076981801,0.7372152728762633,2.032980085917089,0.34437546278749676,0.8147161609747963,3.481756378721718,1.1623089804137197,2.370070818555584,3.659807982849756,0.9417278184540313,0.7768553915352765,0.21040736489619044,3.6831521262903615,0.11930625079423023,1.8764868029500896,1.0069604494716333,0.31912413562892766,0.6529941479585288,0.19739851490104898,3.6198090077300202,0.18748568656443598,3.2830449246490776,0.48007287474688487,0.7971730714062097,0.16780242610081572,0.25121918039199825,0.26502712501732684,0.6644317571689314,1.7193515157681967,0.541968903462887,1.2184696738386838,0.6481011255523961,3.908536819997904,2.549312420133723,0.6881585578916948,0.19791329014506512,0.33109468958027277,1.396764489653576,0.8387816400801184,0.25446463757224624,0.8999686146309105,0.016648102688563395,0.6752305109215452,1.5889690398479057,5.869007443941372,2.28454278916518,0.4149262973438442,3.1315403905432118,0.20063830845197259,0.7946123419349548,0.9012866101038738,0.15788902617878992,0.11841080547570836,2.1971049396190603,1.3986112907571437,1.638091144566374,0.04691126402972,0.0390620295274497,1.3528345623168818,1.1769895991678112,1.7995463981474984,0.19696719091703446,2.9980517192744496,0.1497845472784741,0.8433148294713649,1.8277658966668875,0.21534229086416823,0.15736850846314862,2.003397849079835,0.969864000363037,1.554640679697954,0.05061607440649214]
print(W)
# 求类1的分组情况
NewArray = np.ones((Class1, T+1))
# 第0组
W1 = W[0:14]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 14):
        add1 += W1[j] * X[i, G1[j]]
    NewArray[i][0] = add1
# 第1组
W2 = W[14:29]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 15):
        add2 += W2[j] * X[i, G2[j]]
    NewArray[i][1] = add2
# 第2组
W3 = W[29:44]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 15):
        add3 += W3[j] * X[i, G3[j]]
    NewArray[i][2] = add3

# print(NewArray)

# 求类2的分组情况
NewArray1 = np.ones((Class2, T+1)) * 2
# 第0组
W4 = W[44:58]
for i in range(Class1, n):
    add1 = 0
    for j in range(0, 14):
        add1 += W4[j] * X[i, G1[j]]
    NewArray1[i-Class1][0] = add1
# 第1组
W5 = W[58:73]
for i in range(Class1, n):
    add2 = 0
    for j in range(0, 15):
        add2 += W5[j] * X[i, G2[j]]
    NewArray1[i-Class1][1] = add2
# 第2组
W6 = W[73:88]
for i in range(Class1, n):
    add3 = 0
    for j in range(0, 15):
        add3 += W6[j] * X[i, G3[j]]
    NewArray1[i-Class1][2] = add3
# print(NewArray1)

# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray, NewArray1))
print(NewArray)

# 去掉类标记
NewArray = np.delete(NewArray, T, axis = 1)

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
Pro1 = 147/182
Pro2 = 35/182
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
