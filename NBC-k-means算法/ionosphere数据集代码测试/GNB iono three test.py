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

X = np.loadtxt('[014]ionosphere(0-1).txt')
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
G1 = [6, 13, 14, 24, 30]  # 7
G2 = [5, 7, 9, 17, 21, 25, 31] # 3
G3 = [0, 1, 2, 3, 4, 8, 10, 11, 12, 15, 16, 18, 19, 20, 22, 23, 26, 27, 28, 29]  # 12
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
train_index1 = [113, 10, 26, 3, 87, 101, 34, 117, 105, 79, 22, 112, 104, 76, 115, 92, 55, 111, 60, 38, 29, 59, 53, 54, 123, 57, 93, 28, 52, 97, 80, 78, 90, 35, 16, 68, 13, 48, 82, 72, 15, 47, 9, 51, 30, 1, 75, 32, 17, 70, 18, 88, 84, 96, 41, 74, 5, 24, 8, 58, 33, 100, 64]
val_index1 = [0, 116, 62, 14, 36, 65, 27, 66, 99, 67, 69, 125, 91, 12, 56, 83, 20, 46, 25, 86, 73, 109, 2, 40, 6]
test_index1 = [4, 7, 11, 19, 21, 23, 31, 37, 39, 42, 43, 44, 45, 49, 50, 61, 63, 71, 77, 81, 85, 89, 94, 95, 98, 102, 103, 106, 107, 108, 110, 114, 118, 119, 120, 121, 122, 124]
train_index2 = [187, 191, 172, 36, 107, 167, 175, 143, 129, 70, 202, 25, 161, 48, 222, 141, 160, 67, 18, 162, 140, 134, 40, 156, 193, 32, 188, 154, 168, 163, 111, 19, 93, 63, 82, 87, 182, 95, 203, 194, 147, 206, 9, 137, 61, 122, 217, 165, 209, 46, 89, 120, 76, 177, 101, 51, 49, 117, 151, 148, 21, 68, 138, 58, 224, 127, 26, 62, 71, 213, 110, 27, 199, 30, 190, 17, 92, 152, 11, 142, 205, 3, 136, 185, 86, 220, 208, 59, 196, 197, 52, 35, 91, 39, 119, 2, 5, 121, 181, 123, 170, 6, 179, 53, 16, 192, 212, 173, 133, 184, 88, 72]
val_index2 = [180, 69, 85, 135, 98, 57, 223, 195, 178, 1, 116, 139, 104, 108, 153, 159, 64, 100, 94, 42, 83, 169, 47, 65, 211, 60, 216, 219, 33, 126, 34, 99, 24, 20, 79, 150, 115, 54, 75, 14, 31, 13, 74, 12, 102]
test_index2 = [0, 4, 7, 8, 10, 15, 22, 23, 28, 29, 37, 38, 41, 43, 44, 45, 50, 55, 56, 66, 73, 77, 78, 80, 81, 84, 90, 96, 97, 103, 105, 106, 109, 112, 113, 114, 118, 124, 125, 128, 130, 131, 132, 144, 145, 146, 149, 155, 157, 158, 164, 166, 171, 174, 176, 183, 186, 189, 198, 200, 201, 204, 207, 210, 214, 215, 218, 221]
#W = getRandom(m * K) * 100
W=[0.10255409159443535, 0.05910879449314961, 0.01740187784922999, 0.019835771491749754, 0.10398657640574877, 0.007562109932425305, 0.04854462578407138, 0.36746478947158767, 0.03309148600152904, 0.16810520420720632, 0.05206930462239259, 0.2612663142713397, 0.03109201115888739, 0.9145997104514905, 0.008454065433017827, 0.22480067654406186, 0.23507727677077844, 0.4352846816692928, 0.09374449400327549, 0.1154985205188873, 0.21260899048315812, 0.062430353702482635, 0.5367500067338336, 0.23958651476753018, 0.05781631005204172, 0.30013215463691056, 0.21569890427517643, 0.07508976397972243, 0.13121679359131086, 0.296285927661312, 0.09305627568057978, 0.06709451974352554, 0.04179444123473175, 0.03242596221178968, 0.6441814814243652, 0.06472594065308492, 0.0037226590097975657, 0.18273563977124066, 0.23397963160200191, 0.2402582604477802, 0.10230215059567725, 0.402622508882535, 0.04708015054651166, 0.05576031536705841, 0.1572503720744939, 0.0888726581806622, 0.0005795618911344315, 0.08767261398871043, 0.10303759046349761, 0.06416063861129362, 0.013728802005248106, 0.009020530311459518, 0.034415903796512666, 0.21636214842496376, 0.24782301249397706, 0.672972646908625, 0.25136246123790074, 0.003959169307133812, 0.0005070832019523969, 0.08266843387358815, 0.019147265539920532, 0.0291642526528625, 0.11546360421001997, 0.1629332110973292]




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



