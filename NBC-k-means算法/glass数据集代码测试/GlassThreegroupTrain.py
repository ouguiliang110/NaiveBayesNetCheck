import numpy as np
import math


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    a = 1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro


def getRandom(num):
    Ran = np.random.dirichlet(np.ones(num) * 2, size = 1)
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

X = np.loadtxt('[010]glass(0-1).txt')
# 其中有97
T = 3  # 组数量大小
m = 9  # 属性数量
n = 214  # 样本数目
K = 6  # 类标记数量
Class1 = 70
Class2 = 76
Class3 = 17
Class4 = 29
Class5 = 13
Class6 = 9
# 主要过程：分组
G1 = [6]  # 1
G2 = [1, 2, 3, 5, 7, 8]  # 6
G3 = [0, 4]  # 2

W = getRandom(m * 6) * 10
W=[0.09145301009458211,0.6107913264824597,0.8911516020853948,0.0751546835366887,2.1900835827655474,0.5513515876044378,0.765388218982456,3.0909686940295704,0.06215051965572873,2.0474108243323803,0.17174182366054552,2.6681803283903482,2.4363937643126117,3.774176460042098,2.046222189219981,6.140651421189182,2.0185616970339093,3.3017131552920356,1.1997629302114907,0.002341485440872454,1.8441586348836247,1.4262999914449626,1.0059926869990112,1.490414785451593,0.16377996599416528,2.6111377295849225,4.565373659149633,1.6942386126125963,5.693382274546994,1.856314337008437,0.8345481731807084,0.6666836558865294,2.419848836798031,0.9445189248946202,2.7890797898907516,1.8339266408061234,0.27533636726834726,0.7621831778269325,0.20308686303667092,1.2812300033795496,0.014291005286235002,1.445873443754604,0.01718667141185728,2.2329327648270185,6.0467912963751305,12.27606377353878,0.9138481799771727,1.2111240687657545,3.839194122941589,0.57888469879676,0.2574716439249104,0.10111473941781968,1.0651254099033185,1.502913766072523]
print(W)
# 求类1的分组情况
NewArray1 = np.ones((Class1, T + 1))
# 第0组
W1 = W[0:1]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 1):
        add1 += W1[j] * X[i, G1[j]]
    NewArray1[i][0] = add1
# 第1组
W2 = W[1:7]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 6):
        add2 += W2[j] * X[i, G2[j]]
    NewArray1[i][1] = add2
# 第2组
W3 = W[7:9]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 2):
        add3 += W3[j] * X[i, G3[j]]
    NewArray1[i][2] = add3
# print(NewArray1)

# 求类2的分组情况
NewArray2 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[9:10]
for i in range(Class1, Class1 + Class2):
    add1 = 0
    for j in range(0, 1):
        add1 += W4[j] * X[i, G1[j]]
    NewArray2[i - Class1][0] = add1
# 第1组
W5 = W[10:16]
for i in range(Class1, Class1 + Class2):
    add2 = 0
    for j in range(0, 6):
        add2 += W5[j] * X[i, G2[j]]
    NewArray2[i - Class1][1] = add2
# 第2组
W6 = W[16:18]
for i in range(Class1, Class1 + Class2):
    add3 = 0
    for j in range(0, 2):
        add3 += W6[j] * X[i, G3[j]]
    NewArray2[i - Class1][2] = add3
# print(NewArray2)

# 求类3的分组情况
NewArray3 = np.ones((Class3, T + 1)) * 3
# 第0组
W7 = W[18:19]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add1 = 0
    for j in range(0, 1):
        add1 += W7[j] * X[i, G1[j]]
    NewArray3[i - Class1 - Class2][0] = add1
# 第1组
W8 = W[19:25]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add2 = 0
    for j in range(0, 6):
        add2 += W8[j] * X[i, G2[j]]
    NewArray3[i - Class1 - Class2][1] = add2
# 第2组
W9 = W[25:27]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add3 = 0
    for j in range(0, 2):
        add3 += W9[j] * X[i, G3[j]]
    NewArray3[i - Class1 - Class2][2] = add3
# print(NewArray3)

# 求类4的分组情况
NewArray4 = np.ones((Class4, T + 1)) * 4
# 第0组
W10 = W[27:28]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add1 = 0
    for j in range(0, 1):
        add1 += W10[j] * X[i, G1[j]]
    NewArray4[i - Class1 - Class2 - Class3][0] = add1
# 第1组
W11 = W[28:34]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add2 = 0
    for j in range(0, 6):
        add2 += W11[j] * X[i, G2[j]]
    NewArray4[i - Class1 - Class2 - Class3][1] = add2
# 第2组
W12 = W[34:36]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add3 = 0
    for j in range(0, 2):
        add3 += W12[j] * X[i, G3[j]]
    NewArray4[i - Class1 - Class2 - Class3][2] = add3
# print(NewArray4)
# 求类5的分组情况
NewArray5 = np.ones((Class5, T + 1)) * 5
# 第0组
W13 = W[36:37]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add1 = 0
    for j in range(0, 1):
        add1 += W13[j] * X[i, G1[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][0] = add1
# 第1组
W14 = W[37:43]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add2 = 0
    for j in range(0, 6):
        add2 += W14[j] * X[i, G2[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][1] = add2
# 第2组
W15 = W[43:45]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add3 = 0
    for j in range(0, 2):
        add3 += W15[j] * X[i, G3[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][2] = add3
# print(NewArray5)

# 求类6的分组情况
NewArray6 = np.ones((Class6, T + 1)) * 6
# 第0组
W16 = W[45:46]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add1 = 0
    for j in range(0, 1):
        add1 += W16[j] * X[i, G1[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][0] = add1
# 第1组
W17 = W[46:52]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add2 = 0
    for j in range(0, 6):
        add2 += W17[j] * X[i, G2[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][1] = add2
# 第2组
W18 = W[52:54]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add3 = 0
    for j in range(0, 2):
        add3 += W18[j] * X[i, G3[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][2] = add3
# print(NewArray6)
# 合并两个数组
NewArray = np.vstack((NewArray1, NewArray2, NewArray3, NewArray4, NewArray5, NewArray6))
print(NewArray)

# 去掉类标记
NewArray = np.delete(NewArray, T, axis = 1)
X = NewArray

# 取训练集和测试集7：3比例，70 76 17 29 13 9  训练集149，测试65
trainSet1 = X[0:49, :]  # 49
trainSet2 = X[70:123, :]  # 53
trainSet3 = X[146:158, :]  # 12
trainSet4 = X[163:183, :]  # 20
trainSet5 = X[192:201, :]  # 9
trainSet6 = X[205:211, :]  # 6
# trainingSet = np.vstack((Data1, Data2))
# print(trainingSet)
testSet1 = X[49:70, :]  # 21
testSet2 = X[123:146, :]  # 23
testSet3 = X[158:163, :]  # 5
testSet4 = X[183:192, :]  # 9
testSet5 = X[201:205, :]  # 4
testSet6 = X[211:214, :]  # 3
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
Mean2 = np.mean(trainSet2, axis = 0)
Mean3 = np.mean(trainSet3, axis = 0)
Mean4 = np.mean(trainSet4, axis = 0)
Mean5 = np.mean(trainSet5, axis = 0)
Mean6 = np.mean(trainSet6, axis = 0)

var1 = np.var(trainSet1, axis = 0)
var2 = np.var(trainSet2, axis = 0)
var3 = np.var(trainSet3, axis = 0)
var4 = np.var(trainSet4, axis = 0)
var5 = np.var(trainSet5, axis = 0)
var6 = np.var(trainSet6, axis = 0)
print(var1, var2, var3, var4, var5, var6)

# 先求P(C)
Pro1 = (49 + 1) / (149 + 6)
Pro2 = (53 + 1) / (149 + 6)
Pro3 = (12 + 1) / (149 + 6)
Pro4 = (20 + 1) / (149 + 6)
Pro5 = (9 + 1) / (149 + 6)
Pro6 = (6 + 1) / (149 + 6)
# print(Pro1)
# print(Pro2)

# 统计正确数量和计算准确率
# 计算第一类
add = 0
for i in range(0, 49):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet1[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet1[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(trainSet1[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(trainSet1[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(trainSet1[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(trainSet1[i][j], Mean6[j], var6[j])
    if (Pro1 * sum >= Pro2 * sum1) & (Pro1 * sum >= Pro3 * sum2) & (Pro1 * sum >= Pro4 * sum3) & (
            Pro1 * sum >= Pro5 * sum4) & (Pro1 * sum >= Pro6 * sum5):
        add += 1
    else:
        add += 0
print("第一类正确数量(总数49)：")
print(add)

# 计算第二类
add1 = 0
for i in range(0, 53):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet2[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet2[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(trainSet2[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(trainSet2[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(trainSet2[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(trainSet2[i][j], Mean6[j], var6[j])
    if (Pro2 * sum1 >= Pro1 * sum) & (Pro2 * sum1 >= Pro3 * sum2) & (Pro2 * sum1 >= Pro4 * sum3) & (
            Pro2 * sum1 >= Pro5 * sum4) & (Pro2 * sum1 >= Pro6 * sum5):
        add1 += 1
    else:
        add1 += 0
print("第二类正确数量(总数53)：")
print(add1)

# 计算第三类
add2 = 0
for i in range(0, 12):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet3[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet3[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(trainSet3[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(trainSet3[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(trainSet3[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(trainSet3[i][j], Mean6[j], var6[j])
    if (Pro3 * sum2 >= Pro1 * sum) & (Pro3 * sum2 >= Pro2 * sum1) & (Pro3 * sum2 >= Pro4 * sum3) & (
            Pro3 * sum2 >= Pro5 * sum4) & (Pro3 * sum2 >= Pro6 * sum5):
        add2 += 1
    else:
        add2 += 0
print("第三类正确数量(总数12)：")
print(add2)

add3 = 0
for i in range(0, 20):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet4[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet4[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(trainSet4[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(trainSet4[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(trainSet4[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(trainSet4[i][j], Mean6[j], var6[j])
    if (Pro4 * sum3 >= Pro1 * sum) & (Pro4 * sum3 >= Pro2 * sum1) & (Pro4 * sum3 >= Pro3 * sum2) & (
            Pro4 * sum3 >= Pro5 * sum4) & (Pro4 * sum3 >= Pro6 * sum5):
        add3 += 1
    else:
        add3 += 0
print("第四类正确数量(总数18)：")
print(add3)

add4 = 0
for i in range(0, 9):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet5[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet5[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(trainSet5[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(trainSet5[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(trainSet5[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(trainSet5[i][j], Mean6[j], var6[j])
    if (Pro5 * sum4 >= Pro1 * sum) & (Pro5 * sum4 >= Pro2 * sum1) & (Pro5 * sum4 >= Pro3 * sum2) & (
            Pro5 * sum4 >= Pro4 * sum3) & (Pro5 * sum4 >= Pro6 * sum5):
        add4 += 1
    else:
        add4 += 0
print("第五类正确数量(总数8)：")
print(add4)

add5 = 0
for i in range(0, 6):
    sum = 1
    for j in range(0, T):
        sum *= getPro(trainSet6[i][j], Mean1[j], var1[j])
    sum1 = 1
    for j in range(0, T):
        sum1 *= getPro(trainSet6[i][j], Mean2[j], var2[j])
    sum2 = 1
    for j in range(0, T):
        sum2 *= getPro(trainSet6[i][j], Mean3[j], var3[j])
    sum3 = 1
    for j in range(0, T):
        sum3 *= getPro(trainSet6[i][j], Mean4[j], var4[j])
    sum4 = 1
    for j in range(0, T):
        sum4 *= getPro(trainSet6[i][j], Mean5[j], var5[j])
    sum5 = 1
    for j in range(0, T):
        sum5 *= getPro(trainSet6[i][j], Mean6[j], var6[j])
    if (Pro6 * sum5 >= Pro1 * sum) & (Pro6 * sum5 >= Pro2 * sum1) & (Pro6 * sum5 >= Pro3 * sum2) & (
            Pro6 * sum5 >= Pro4 * sum3) & (Pro6 * sum5 >= Pro5 * sum4):
        add5 += 1
    else:
        add5 += 0
print("第六类正确数量(总数6)：")
print(add5)

# 准确率
print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5) / 149))
