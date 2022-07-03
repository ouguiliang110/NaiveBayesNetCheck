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
Class1=97
Class2=111
# 随机产生多少个和为1的随机数W
G1=[15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 33, 34, 35, 36]
G2=[5, 14, 23, 26, 27, 28, 29, 30, 31, 37, 38, 39]
G3=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
W=getRandom(m*2)*100
#求类1的分组情况
NewArray = np.ones((97, 4))
W=[1.102619747610236,0.4530334024580028,0.461294608030703,0.4741356930246272,0.230109581043747,0.28094105984804374,1.0009759354574412,0.011695825052220152,0.9646660050625215,0.06412962661505321,1.6293127153063156,0.3099763091968231,0.8203285687959565,2.902796024026305,0.31375685656240004,0.5569007524799782,0.3054367618968766,1.2442498139497569,0.07835053211468349,0.24551664290065803,0.1934766737025846,0.24082982856456986,0.031968301626633014,2.493352716452128,1.5898295934891373,1.1947609473649483,0.933442704577375,1.3036356526594457,1.0256672736168109,0.07955362167047594,3.8494488747181186,0.773084386482142,0.5897264033654013,1.594700445902111,0.9961958964646328,0.7455226731873997,0.3313184463552222,1.5993145851178943,0.3904474078366872,0.26151622005423497,0.26575569982780484,2.294573699257156,1.4674577521174483,0.4826182067837766,1.1417083457250596,0.04767771704829827,1.9091564162634336,0.14981762567568874,0.6321420154689257,0.3828484598893967,0.11968737781018456,1.2095074144540336,0.006675123161833364,1.0684531524795973,0.3889702873557503,0.1358040010435035,1.9696647053917,0.10146998769515805,0.3765045680259296,0.7110715504316636,0.8668429555058884,0.15809711078108593,0.215004611653182,0.12525462005542096,0.9644025257824346,0.12831698512831513,0.5078125980636884,0.6700274234109519,0.6784426582674145,0.4617948022827024,0.666895762144236,0.1821536723442845,0.34516719715647515,3.1567626670255082,0.40700578981886293,0.6434451717240888,2.2107196622087875,2.819654143976251,2.3669096468493254,0.21013868652833947,0.4451223423760275,0.12757041117354337,0.4357143010710127,1.5515141712206546,0.4118420978746486,4.783190059780515,0.8393939336588037,0.6830680831492031,0.6883679277391628,0.882722091865201,0.174471731220198,0.04376357848599366,0.16408058608938886,0.6583245016041258,0.09148210595135339,0.1053985778154023,0.11001506163376333,0.010693926758787036,0.40090187197910804,0.11601989546211776,0.07362472838699932,2.2143855909922263,1.6331686240999481,2.360556183701671,0.07033376717599393,6.9844964492559,1.0338684344965463,0.7035597953305401,0.8654048778925805,0.3684940249461568,0.2500331081914778,0.9767520567194787,0.17733071663060618,0.11380092030445735,0.1630078283709024,0.030656595626540818,1.7770867843383564,0.147383564024807,0.8803806916006154,0.13758608975128955]
# 第0组
W1 = W[0:14]
for i in range(0, 97):
    add1 = 0
    for j in range(0,14):
        add1+=W1[j] * X[i, G1[j]]
    NewArray[i][0]=add1
# 第1组
W2 = W[14:26]
for i in range(0, 97):
    add2 = 0
    for j in range(0,12):
        add2+=W2[j] * X[i, G2[j]]
    NewArray[i][1]=add2
# 第2组
W3 = W[26:60]
for i in range(0, 97):
    add3 = 0
    for j in range(0,34):
        add3+=W3[j] * X[i, G3[j]]
    NewArray[i][2]=add3

#print(NewArray)

#求类2的分组情况
NewArray1 = np.ones((111, 4))*2
# 第0组
W4=W[60:74]
for i in range(Class1, n):
    add1 = 0
    for j in range(0,14):
        add1+=W1[j] * X[i, G1[j]]
    NewArray1[i-Class1][0]=add1
# 第1组
W5=W[74:86]
for i in range(Class1, n):
    add2 = 0
    for j in range(0,12):
        add2+=W2[j] * X[i, G2[j]]
    NewArray1[i-Class1][1]=add2
# 第2组
W6=W[86:120]
for i in range(Class1, n):
    add3 = 0
    for j in range(0,34):
        add3+=W3[j] * X[i, G3[j]]
    NewArray1[i-Class1][2]=add3

#print(NewArray1)

#合并两个数组，得到真正的合并数据结果
NewArray=np.vstack((NewArray,NewArray1))
print(NewArray)



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
    for j in range(0, 3):
        sum *= getPro(Data1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, 3):
        sum1 *= getPro(Data1[i][j], Mean2[j], var2[j])
    if Pro1*sum > Pro2*sum1:
        add += 1
    elif Pro1*sum < Pro2*sum1:
        add += 0
print("第一类正确数量(总数27)：")
print(add)
add1 = 0
for i in range(0, 34):
    sum = 1
    for j in range(0, 3):
        sum *= getPro(Data2[i][j], Mean2[j], var2[j])
    sum1 = 1
    for j in range(0, 3):
        sum1 *= getPro(Data2[i][j], Mean1[j], var1[j])
    if Pro2*sum > Pro1*sum1:
        add1 += 1
    elif Pro2*sum < Pro1*sum1:
        add1 += 0
print("第二类正确数量(总数34)：")
print(add1)
# 准确率
print("accuracy:{:.2%}".format((add + add1) / 61))

