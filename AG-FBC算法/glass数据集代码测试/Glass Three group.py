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
clf = GaussianNB()
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
W=[0.10747686537920374,0.24684797012189214,0.002656213195784278,0.26294153556243494,0.3320799220541762,0.1913940002437224,0.06723177349200121,0.0012389549810432087,0.1743723081239652,0.04062635457809563,0.004693432628565464,0.0512687900557442,0.07806580534436121,0.2915160003829332,0.11345196125972885,0.06859294806312738,0.1933401415717441,0.05698022954966761,0.20396632474819631,0.17039827940489022,0.015656442295012712,0.010852428361965465,0.23717992261826484,0.23997784407793207,0.507314550973968,0.6774490219747976,0.09963664861077316,0.18159261612907543,0.7656202439021532,0.01596348008680077,0.03230219391187042,0.334582957389276,0.09474262893847674,0.6022907890195066,0.05642924044292305,0.045229775725333037,0.01948270131780183,0.08398177892056244,0.0009238652578891899,0.7620824438884598,0.08499851719247291,0.30352816637288904,0.04383632737315395,0.23029744634260912,0.035882939732308244,0.4277332680452517,0.20146125177512475,0.19210155760193165,0.09458006046640371,0.5689097991381232,0.07734984934436624,0.019743231818431555,0.2701998983450463,0.006946301867768319]
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
print(NewArray.shape)
# 去掉类标记
Y=NewArray[:,T]
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


# 准确率
clf.fit(X,Y)
C1=clf.predict(testSet1)
add=sum(C1==1)
print(add)
C2=clf.predict(testSet2)
add1=sum(C2==2)
print(add1)
C3=clf.predict(testSet3)
add2=sum(C3==3)
print(add2)
C4=clf.predict(testSet4)
add3=sum(C4==4)
print(add3)
C5=clf.predict(testSet5)
add4=sum(C5==5)
print(add4)
C6=clf.predict(testSet6)
add5=sum(C6==6)
print(add5)
print("accuracy:{:.2%}".format((add+add1+add2+add3+add4+add5)/65))
