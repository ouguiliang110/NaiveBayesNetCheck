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

X = np.loadtxt('[008]band(0-1).txt')
# 其中有97
m = X.shape[1] - 1  # 属性数量
n = X.shape[0]  # 样本数目
T = 3
K = 2  # 类标记数量
# 主要过程：分组
# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 随机产生多少个和为1的随机数W
G1 = [1, 2, 3, 5] # 5
G2 = [4, 6, 8, 9, 10, 11, 13, 15, 16, 17, 20, 21, 22]# 7
G3 = [0, 7, 12, 14, 18, 19]# 20
len1=len(G1)
len2=len(G2)
len3=len(G3)
split=len1+len2+len3
#随机训练集，验证集，测试集区

idx = np.random.choice(np.arange(Class1), size = train1, replace = False)
train_index1 = np.array(idx)
val_index1 = np.random.choice(np.delete(np.arange(Class1), train_index1), size = val1, replace = False)
test_index1 = np.delete(np.arange(Class1), np.append(train_index1, val_index1))

idx1 = np.random.choice(np.arange(Class2), size = train2, replace = False)
train_index2 = np.array(idx1)
val_index2 = np.random.choice(np.delete(np.arange(Class2), train_index2), size = val2, replace = False)
test_index2 = np.delete(np.arange(Class2), np.append(train_index2, val_index2))

print("train_index1 =",list(train_index1))
print("val_index1 =",list(val_index1))
print("test_index1 =",list(test_index1))
print("train_index2 =",list(train_index2))
print("val_index2 =",list(val_index2))
print("test_index2 =",list(test_index2))




#确认训练集，验证集，测试集区
train_index1 = [165, 255, 246, 137, 172, 185, 95, 11, 307, 23, 310, 121, 190, 19, 295, 75, 180, 234, 215, 275, 24, 134, 289, 224, 25, 309, 199, 166, 277, 229, 13, 52, 285, 108, 91, 113, 238, 192, 267, 109, 186, 308, 296, 138, 86, 280, 194, 283, 262, 85, 161, 214, 210, 126, 141, 139, 12, 269, 168, 58, 259, 163, 230, 226, 30, 157, 22, 127, 164, 107, 132, 133, 27, 253, 66, 254, 114, 181, 57, 198, 222, 4, 284, 44, 288, 99, 184, 174, 260, 158, 135, 47, 16, 150, 272, 112, 26, 82, 42, 232, 55, 291, 142, 178, 261, 60, 148, 73, 46, 17, 36, 92, 40, 130, 237, 8, 216, 88, 143, 77, 18, 131, 197, 248, 264, 9, 175, 244, 37, 306, 211, 151, 34, 83, 62, 145, 56, 7, 299, 276, 87, 76, 98, 110, 300, 303, 80, 123, 257, 81, 45, 200, 292, 129, 206, 136]
val_index1 = [217, 159, 51, 171, 128, 252, 298, 160, 281, 152, 282, 189, 59, 90, 96, 70, 270, 119, 69, 203, 74, 71, 3, 15, 223, 294, 271, 286, 125, 193, 140, 49, 187, 245, 278, 304, 268, 79, 195, 227, 201, 10, 115, 167, 20, 249, 50, 231, 274, 302, 154, 28, 106, 225, 290, 61, 146, 64, 43, 35, 250, 93]
test_index1 = [0, 1, 2, 5, 6, 14, 21, 29, 31, 32, 33, 38, 39, 41, 48, 53, 54, 63, 65, 67, 68, 72, 78, 84, 89, 94, 97, 100, 101, 102, 103, 104, 105, 111, 116, 117, 118, 120, 122, 124, 144, 147, 149, 153, 155, 156, 162, 169, 170, 173, 176, 177, 179, 182, 183, 188, 191, 196, 202, 204, 205, 207, 208, 209, 212, 213, 218, 219, 220, 221, 228, 233, 235, 236, 239, 240, 241, 242, 243, 247, 251, 256, 258, 263, 265, 266, 273, 279, 287, 293, 297, 301, 305, 311]
train_index2 = [8, 55, 158, 150, 12, 128, 222, 186, 9, 200, 26, 111, 178, 47, 19, 87, 83, 124, 58, 103, 208, 37, 95, 66, 159, 3, 62, 72, 56, 84, 183, 30, 129, 114, 122, 167, 115, 99, 50, 81, 105, 119, 74, 164, 138, 42, 207, 118, 196, 7, 224, 70, 162, 35, 11, 209, 53, 59, 75, 112, 134, 44, 16, 165, 198, 78, 176, 146, 116, 133, 64, 210, 14, 144, 221, 185, 213, 32, 43, 39, 93, 217, 0, 54, 194, 90, 180, 126, 25, 80, 212, 28, 140, 100, 36, 154, 152, 226, 161, 218, 4, 168, 225, 113, 137, 174, 169, 29, 61, 82, 223, 71, 104, 51]
val_index2 = [214, 60, 85, 182, 123, 215, 192, 197, 67, 193, 41, 63, 143, 20, 132, 220, 147, 190, 68, 155, 102, 211, 17, 101, 189, 2, 179, 76, 108, 163, 94, 151, 136, 142, 170, 148, 171, 160, 172, 1, 204, 46, 10, 187, 88]
test_index2 = [5, 6, 13, 15, 18, 21, 22, 23, 24, 27, 31, 33, 34, 38, 40, 45, 48, 49, 52, 57, 65, 69, 73, 77, 79, 86, 89, 91, 92, 96, 97, 98, 106, 107, 109, 110, 117, 120, 121, 125, 127, 130, 131, 135, 139, 141, 145, 149, 153, 156, 157, 166, 173, 175, 177, 181, 184, 188, 191, 195, 199, 201, 202, 203, 205, 206, 216, 219, 227]

#W = getRandom(m * K) * 100
W=[0.035499899981389055, 0.6336933451051403, 0.03923440607731665, 0.18026218845604594, 0.22013478701611883, 0.003525052629063542, 0.012493839007439396, 0.1883693334700058, 0.9808520503203509, 0.7896295837350034, 0.052639421513409584, 0.01890604975020231, 0.32831844382797104, 0.6864082838719081, 0.056614227740222976, 0.04591474021141705, 0.1025573700425145, 0.40458824398043636, 0.3000926773543627, 0.3872882308481803, 0.5913392895623597, 0.028627110088752242, 0.15231361681506914, 0.04358833397049673, 0.40395563807415985, 0.16883975587386735, 0.0886563741299139, 0.05423977063441135, 0.001971327556271418, 0.04386326605918679, 0.46669854380044123, 0.7343447202738261, 0.01662362958126283, 0.29921505207222593, 0.3271960700661158, 0.18567869761865502, 0.12650212180878379, 0.019786438913151055, 0.2151897754261512, 0.03796701942069034, 0.009083497980315831, 0.06246272502089353, 0.1805198520523026, 0.06004643094033695, 0.13715286717287664, 0.07711590014898187]








NewArray1 = np.ones((Class1, T + 1))
# 第0组
W1 = W[0:len1]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, len1):
        add1 += W1[j] * X[i, G1[j]]
    NewArray1[i][0] = add1
# 第1组
W2 = W[len1:len1+len2]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, len2):
        add2 += W2[j] * X[i, G2[j]]
    NewArray1[i][1] = add2
# 第2组
W3 = W[len1+len2:len1+len2+len3]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, len3):
        add3 += W3[j] * X[i, G3[j]]
    NewArray1[i][2] = add3
# print(NewArray1)

# 求类2的分组情况
NewArray2 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[split:split+len1]
for i in range(Class1, Class1 + Class2):
    add1 = 0
    for j in range(0, len1):
        add1 += W4[j] * X[i, G1[j]]
    NewArray2[i - Class1][0] = add1
# 第1组
W5 = W[split+len1:split+len1+len2]
for i in range(Class1, Class1 + Class2):
    add2 = 0
    for j in range(0, len2):
        add2 += W5[j] * X[i, G2[j]]
    NewArray2[i - Class1][1] = add2
# 第2组
W6 = W[split+len1+len2:split+len1+len2+len3]
for i in range(Class1, Class1 + Class2):
    add3 = 0
    for j in range(0, len3):
        add3 += W6[j] * X[i, G3[j]]
    NewArray2[i - Class1][2] = add3
# print(NewArray2)

# print(NewArray1)
# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray1, NewArray2))
# print(NewArray)

# 随机抽取样本训练集和测试集样本

X1 = NewArray[0:Class1, :]
X2 = NewArray[Class1:Class1 + Class2, :]

Data1 = X1[train_index1, :]
Data2 = X2[train_index2, :]
trainSet=np.vstack((Data1,Data2))
Y=trainSet[:,T]
trainSet=np.delete(trainSet,T,axis = 1)

testSet1 = np.delete(X1[test_index1, :], T, axis = 1)
testSet2 = np.delete(X2[test_index2, :], T, axis = 1)
trainSet1 = np.delete(Data1, T, axis = 1)
trainSet2 = np.delete(Data2, T, axis = 1)
valSet1=np.delete(X1[val_index1,:],T,axis = 1)
valSet2=np.delete(X2[val_index2,:],T,axis = 1)

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
Mean2 = np.mean(trainSet2, axis = 0)
# print(Mean2)
var1 = np.var(trainSet1, axis = 0)
var2 = np.var(trainSet2, axis = 0)



clf=GaussianNB()

clf.fit(trainSet,Y)

C1 = clf.predict(testSet1)
add = sum(C1 == 1)
print("第一类正确数量(总数):", test1)
print(add)
C2 = clf.predict(testSet2)
add1 = sum(C2 == 2)
print("第二类正确数量(总数)：", test2)
print(add1)

print("accuracy:{:.2%}".format((add + add1) / (test1+test2)))



