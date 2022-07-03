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
W=[0.2864498404119782,0.05823978021666256,0.7992322616900005,0.08291983413330496,0.1754968492058913,0.02194008501619485,0.05626526555566957,0.16388222066950747,0.038953719907481484,0.13928661941732956,0.1587516567044353,0.020111871717583604,0.1911938401896917,0.2428553777954161,0.052831613250978106,0.05279727095236011,0.021351148574221376,0.011750441352831728,0.12623274544548704,0.14145470830449217,0.00031266370644126017,0.024921805218063968,0.09169552633150362,0.040380826039638815,0.0020875301963244938,0.13552106557310137,0.040956400374930695,0.18002443573210433,0.14644611961537007,0.018999262850776646,0.18945001263922534,0.01168328664145587,0.034201896508142475,0.10528859882041136,0.06721033883928262,0.15037519944225425,0.00971695842772175,0.05247832383439055,0.010632353705080658,0.0021102980880059606,0.203968482816763,0.056521250382021195,0.18553333246828255,0.012773789761017292,0.01958643757161146,0.170670691327641,0.0812222614425842,0.023420320388090124,0.06349951018312974,0.04527898529191092,0.28086010389181326,0.10998947854270259,0.01640198627370171,0.11052085072575712,0.08179796243978435,0.04356913634397255,0.11543192577564393,0.029359345729574358,0.1309924922163562,0.5994846356764671,0.14625686287290285,0.00854582164300732,0.06215471427509968,0.13970988718633395,0.28892314359306365,0.0542715323628282,0.5833076279990066,0.0675198869916092,0.10747472691116466,0.06741763809403736,0.018130302671458935,0.08613885708965917,0.22444483074720073,0.22244969863989159,0.008657843594726378,0.0057715886120246265,0.3581966220341858,0.09022949202180905,0.08450151033755285,0.26275247957136794,0.1923136191261514,0.021970632468919716,0.0808520981443908,0.0016154399641948023,0.09818388211537935,0.09852825959064922,0.006805161090420961,0.07550280790639409]
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

# 取训练集和测试集5；2：3比例
trainSet1=NewArray[0:106, :]
trainSet2=NewArray[212:239,:]
valSet1=NewArray[106:148,:]
valSet2=NewArray[239:250,:]
# print(trainingSet)
testSet1=NewArray[148:212, :]
testSet2=NewArray[250:267, :]
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
Mean2 = np.mean(trainSet2, axis = 0)
var1 = np.var(trainSet1, axis = 0)
var2 = np.var(trainSet2, axis = 0)

# 先求P(C) 根据机器学习课本中的拉普拉斯修正法
Pro1 =106 / 133
Pro2 = 27 / 133
# print(Pro1)
# print(Pro2)

# 本次代码主要内容是这个，求P(Ai|C)

# 通过朴素贝叶斯算法得到分类器的准确率
add = 0
for i in range(0, 106):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet1[i][j], Mean2[j], var2[j])
    if Pro1 * sum >= Pro2 * sum1:
        add += 1
    elif Pro1 * sum < Pro2 * sum1:
        add += 0
print("第一类正确数量(总数65)：")
print(add)
add1 = 0
for i in range(0,27):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet2[i][j], Mean2[j], var2[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet2[i][j], Mean1[j], var1[j])
    if Pro2 * sum >= Pro1 * sum1:
        add1 += 1
    elif Pro2 * sum < Pro1 * sum1:
        add1 += 0
print("第二类正确数量(总数17)：")
print(add1)
# 准确率
print("accuracy:{:.2%}".format((add + add1) / 133))
