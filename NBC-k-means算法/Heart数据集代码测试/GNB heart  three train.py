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

X = np.loadtxt('[012]heart(0-1).txt')
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
G1 = [1, 5, 6, 8, 9, 11, 12] # 5
G2 = [2, 3, 7, 10]# 7
G3 = [0, 4]# 20
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
train_index1 = [119, 20, 56, 75, 10, 115, 21, 64, 54, 17, 57, 92, 82, 98, 8, 102, 33, 4, 116, 99, 84, 11, 91, 40, 104, 88, 36, 14, 45, 59, 67, 22, 107, 16, 29, 26, 118, 117, 71, 7, 27, 81, 61, 93, 111, 2, 77, 110, 112, 46, 94, 113, 31, 79, 80, 90, 15, 6, 76, 51]
val_index1 = [101, 25, 50, 30, 49, 35, 72, 97, 66, 39, 44, 100, 47, 78, 60, 108, 24, 9, 43, 63, 41, 62, 55, 5]
test_index1 = [0, 1, 3, 12, 13, 18, 19, 23, 28, 32, 34, 37, 38, 42, 48, 52, 53, 58, 65, 68, 69, 70, 73, 74, 83, 85, 86, 87, 89, 95, 96, 103, 105, 106, 109, 114]
train_index2 = [16, 40, 28, 71, 57, 99, 91, 44, 126, 1, 148, 145, 36, 64, 116, 110, 106, 55, 88, 120, 83, 115, 48, 139, 86, 102, 84, 3, 81, 79, 54, 13, 136, 98, 51, 103, 95, 97, 89, 128, 114, 49, 133, 77, 96, 65, 39, 69, 47, 132, 15, 43, 21, 42, 41, 19, 87, 130, 26, 138, 8, 17, 10, 56, 92, 24, 82, 35, 53, 20, 31, 23, 22, 32, 100]
val_index2 = [46, 149, 121, 107, 137, 134, 29, 101, 45, 147, 38, 122, 80, 67, 14, 93, 4, 111, 50, 66, 108, 34, 33, 60, 129, 119, 90, 18, 63, 30]
test_index2 = [0, 2, 5, 6, 7, 9, 11, 12, 25, 27, 37, 52, 58, 59, 61, 62, 68, 70, 72, 73, 74, 75, 76, 78, 85, 94, 104, 105, 109, 112, 113, 117, 118, 123, 124, 125, 127, 131, 135, 140, 141, 142, 143, 144, 146]


W = getRandom(m * K) * 100
W=[0.08307095858934023, 0.41062139427866684, 0.029985051589298158, 0.2599252863031907, 0.3330819614720059, 0.12478710697800821, 0.3730752447830485, 0.18953545672528202, 0.28746742552810134, 0.41906496021945455, 0.46902172575261414, 0.38795135050542445, 0.3778077773495867, 0.03681730899341294, 1.0949765874962039, 0.18487996308285834, 0.05867774224557261, 0.13912944275294759, 0.09033589836097554, 0.009814633076468873, 0.04171644089800296, 0.7121265591448568, 1.7452444191986656, 0.7745203124354788, 0.8706541814893889, 0.49571081075114554]















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

C1 = clf.predict(trainSet1)
add = sum(C1 == 1)
print("第一类正确数量(总数):", train1)
print(add)
C2 = clf.predict(trainSet2)
add1 = sum(C2 == 2)
print("第二类正确数量(总数)：", train2)
print(add1)

print("accuracy:{:.2%}".format((add + add1) / (train1+train2)))



