import numpy as np
import math
from sklearn.naive_bayes import GaussianNB

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

X = np.loadtxt('[028]wineQR(0-1).txt')
# 其中有97
T = 3  # 组数量大小
m = 11  # 属性数量
n = 1599  # 样本数目
K = 6  # 类标记数量
Class1 = 10
Class2 = 53
Class3 = 681
Class4 = 638
Class5 = 199
Class6 = 18
# 主要过程：分组
G1 = [2, 7, 8, 10]  # 4
G2 = [3, 4, 5, 6, 9]  # 5
G3 = [0, 1]  # 2

W = getRandom(m * 6) * 10
W = [0.001330522303841501,0.007976722185791919,0.025096899182677584,0.00935181186423478,0.0019323952409461603,0.034270358729882384,0.030825768993725005,0.036437388256768274,0.020157972125068202,0.022465062523226973,0.018909223982305705,0.004434209962362611,0.00937488375435969,0.031091834763906208,0.009541934367455469,0.0218603123459994,0.03750391344321727,0.034661997736692804,0.004377365450858486,0.011239321361488142,0.022955223502669878,0.00664789465395738,0.009056439960288393,0.0003810633924851946,0.0054368394929133436,0.00526711207581584,0.026284789418424208,0.000345036488334718,0.02500834001707906,0.008939439946800425,0.0027847316100255023,0.001965932366997423,0.017119198663195147,0.01629217698164693,0.005696582286884599,0.009579369563905519,0.014726794742582624,0.00026385009395302943,0.012275990484306222,0.0039024545002301937,0.04262932019637773,0.06942947533881408,0.0007259955412646152,0.005785260841226808,0.0064208471651701704,0.0189879819035395,0.0002351376465478723,0.0005092028653776903,0.04659045661306349,0.0035134565914523297,0.0062736610080556775,0.059038688519131376,0.00517445249164123,0.0001905056073210274,0.0037133309033607455,0.034731307355622826,0.02391554199468359,0.00536373958937519,0.012064166257876216,0.007317830520581392,0.011804245642712447,0.0015161578903044916,0.006779454490645261,0.009511203686070495,0.0281756688129945,0.0218337517094854]
print(W)
# 求类1的分组情况
NewArray1 = np.ones((Class1, T + 1))
# 第0组
W1 = W[0:4]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 4):
        add1 += W1[j] * X[i, G1[j]]
    NewArray1[i][0] = add1
# 第1组
W2 = W[4:9]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 5):
        add2 += W2[j] * X[i, G2[j]]
    NewArray1[i][1] = add2
# 第2组
W3 = W[9:11]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 2):
        add3 += W3[j] * X[i, G3[j]]
    NewArray1[i][2] = add3
# print(NewArray1)

# 求类2的分组情况
NewArray2 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[11:15]
for i in range(Class1, Class1 + Class2):
    add1 = 0
    for j in range(0, 4):
        add1 += W4[j] * X[i, G1[j]]
    NewArray2[i - Class1][0] = add1
# 第1组
W5 = W[15:20]
for i in range(Class1, Class1 + Class2):
    add2 = 0
    for j in range(0, 5):
        add2 += W5[j] * X[i, G2[j]]
    NewArray2[i - Class1][1] = add2
# 第2组
W6 = W[20:22]
for i in range(Class1, Class1 + Class2):
    add3 = 0
    for j in range(0, 2):
        add3 += W6[j] * X[i, G3[j]]
    NewArray2[i - Class1][2] = add3
# print(NewArray2)

# 求类3的分组情况
NewArray3 = np.ones((Class3, T + 1)) * 3
# 第0组
W7 = W[22:26]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add1 = 0
    for j in range(0, 4):
        add1 += W7[j] * X[i, G1[j]]
    NewArray3[i - Class1 - Class2][0] = add1
# 第1组
W8 = W[26:31]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add2 = 0
    for j in range(0, 5):
        add2 += W8[j] * X[i, G2[j]]
    NewArray3[i - Class1 - Class2][1] = add2
# 第2组
W9 = W[31:33]
for i in range(Class1 + Class2, Class1 + Class2 + Class3):
    add3 = 0
    for j in range(0, 2):
        add3 += W9[j] * X[i, G3[j]]
    NewArray3[i - Class1 - Class2][2] = add3
# print(NewArray3)

# 求类4的分组情况
NewArray4 = np.ones((Class4, T + 1)) * 4
# 第0组
W10 = W[33:37]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add1 = 0
    for j in range(0, 4):
        add1 += W10[j] * X[i, G1[j]]
    NewArray4[i - Class1 - Class2 - Class3][0] = add1
# 第1组
W11 = W[37:42]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add2 = 0
    for j in range(0, 5):
        add2 += W11[j] * X[i, G2[j]]
    NewArray4[i - Class1 - Class2 - Class3][1] = add2
# 第2组
W12 = W[42:44]
for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
    add3 = 0
    for j in range(0, 2):
        add3 += W12[j] * X[i, G3[j]]
    NewArray4[i - Class1 - Class2 - Class3][2] = add3
# print(NewArray4)
# 求类5的分组情况
NewArray5 = np.ones((Class5, T + 1)) * 5
# 第0组
W13 = W[44:48]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add1 = 0
    for j in range(0, 4):
        add1 += W13[j] * X[i, G1[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][0] = add1
# 第1组
W14 = W[48:53]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add2 = 0
    for j in range(0, 5):
        add2 += W14[j] * X[i, G2[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][1] = add2
# 第2组
W15 = W[53:55]
for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
    add3 = 0
    for j in range(0, 2):
        add3 += W15[j] * X[i, G3[j]]
    NewArray5[i - Class1 - Class2 - Class3 - Class4][2] = add3
# print(NewArray5)

# 求类6的分组情况
NewArray6 = np.ones((Class6, T + 1)) * 6
# 第0组
W16 = W[55:59]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add1 = 0
    for j in range(0, 4):
        add1 += W16[j] * X[i, G1[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][0] = add1
# 第1组
W17 = W[59:64]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add2 = 0
    for j in range(0, 5):
        add2 += W17[j] * X[i, G2[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][1] = add2
# 第2组
W18 = W[64:66]
for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
    add3 = 0
    for j in range(0, 2):
        add3 += W18[j] * X[i, G3[j]]
    NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][2] = add3
# print(NewArray6)
# 合并两个数组
NewArray = np.vstack((NewArray1, NewArray2, NewArray3, NewArray4, NewArray5, NewArray6))
print(NewArray)
print(NewArray.shape)
Y=NewArray[:,T]
# 去掉类标记
NewArray = np.delete(NewArray, T, axis = 1)
X = NewArray
# 取训练集和测试集7：3比例，10,53,681,638,199,18  训练集1118，测试481
trainSet1 = X[0:7, :]  # 7
trainSet2 = X[10:47, :]  # 37
trainSet3 = X[63:540, :]  # 477
trainSet4 = X[744:1190, :]  # 446
trainSet5 = X[1382:1521, :]  # 139
trainSet6 = X[1581:1593, :]  # 12
# trainingSet = np.vstack((Data1, Data2))
# print(trainingSet)
testSet1 = X[7:10, :]  # 3
testSet2 = X[47:63, :]  # 16
testSet3 = X[540:744, :]  # 204
testSet4 = X[1190:1382, :]  # 192
testSet5 = X[1521:1581, :]  # 60
testSet6 = X[1593:1599, :]  # 6
# testSet = np.vstack((testSet1, testSet2))
# print(testSet)
clf = GaussianNB()
clf.fit(NewArray, Y)
C1 = clf.predict(trainSet1)
add = sum(C1 == 1)
# print(add)
C2 = clf.predict(trainSet2)
add1 = sum(C2 == 2)
# print(add1)
C3 = clf.predict(trainSet3)
add2 = sum(C3 == 3)
# print(add2)
C4 = clf.predict(trainSet4)
add3 = sum(C4 == 4)
# print(add3)
C5 = clf.predict(trainSet5)
add4 = sum(C5 == 5)
# print(add4)
C6 = clf.predict(trainSet6)
add5 = sum(C6 == 6)
# print(add5)
print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5) / 1118))


