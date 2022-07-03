import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

X = np.loadtxt('[021]parkinsons(0-1).txt')
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
G1 = [0, 2, 8, 10, 13, 16, 17]  # 7
G2 = [4, 12, 14]  # 3
G3 = [1, 3, 5, 6, 7, 9, 11, 15, 18, 19, 20, 21]  # 12

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
train_index1 = [45, 68, 140, 99, 21, 2, 5, 32, 18, 74, 39, 20, 79, 125, 94, 34, 115, 109, 136, 9, 129, 138, 4, 62, 55, 127, 77, 31, 133, 36, 76, 117, 38, 85, 54, 10, 35, 96, 57, 142, 144, 33, 130, 75, 100, 118, 47, 135, 69, 114, 46, 139, 101, 50, 58, 90, 1, 105, 11, 51, 137, 25, 93, 70, 88, 28, 104, 119, 23, 59, 89, 53, 44]
val_index1 = [13, 102, 73, 30, 7, 91, 66, 12, 82, 112, 86, 111, 97, 27, 52, 67, 123, 143, 81, 41, 126, 145, 6, 64, 98, 83, 106, 84, 95]
test_index1 = [0, 3, 8, 14, 15, 16, 17, 19, 22, 24, 26, 29, 37, 40, 42, 43, 48, 49, 56, 60, 61, 63, 65, 71, 72, 78, 80, 87, 92, 103, 107, 108, 110, 113, 116, 120, 121, 122, 124, 128, 131, 132, 134, 141, 146]
train_index2 = [40, 45, 27, 46, 29, 34, 4, 3, 2, 23, 15, 21, 11, 0, 31, 28, 32, 8, 18, 20, 42, 6, 38, 24]
val_index2 = [10, 36, 44, 39, 35, 26, 47, 7, 22]
test_index2 = [1, 5, 9, 12, 13, 14, 16, 17, 19, 25, 30, 33, 37, 41, 43]




W = getRandom(m * K) * 100
W=[0.49391207194652226, 0.06701034112691243, 0.0939462025306337, 0.5050201157374029, 0.053841110930498086, 0.19040110531440474, 0.048413174512861715, 0.44757214730333106, 0.19764239351490545, 0.11338224860254832, 0.11319690953604361, 0.2903200617023341, 0.14413933290902298, 0.024782959375860718, 0.10101204971163381, 0.211160528170371, 1.1619761002475781, 0.01829979322090902, 0.029028776751019312, 0.14946950222514194, 0.7144617430240148, 0.09038954338773006, 0.11645519957774111, 0.1903889164378073, 0.09484857382417694, 0.012456284876251783, 0.03707189964420932, 0.09014384740422893, 0.011091273505662249, 0.10738551977354738, 0.02773123551044131, 0.11807560560665085, 0.13012008633985095, 0.20100684394215246, 0.06020942069714693, 0.16595062376663972, 0.009663844215810465, 1.0791389920397598, 0.10334597113060418, 0.9777141724517506, 0.07374250618798797, 0.5993960963593532, 0.428606282645425, 0.10607859228112129]


# 求类1的分组情况
NewArray = np.ones((Class1, T + 1))
# 第0组
W1 = W[0:7]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 7):
        add1 += W1[j] * X[i, G1[j]]
    NewArray[i][0] = add1
# 第1组
W2 = W[7:10]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 3):
        add2 += W2[j] * X[i, G2[j]]
    NewArray[i][1] = add2
# 第2组
W3 = W[10:22]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 12):
        add3 += W3[j] * X[i, G3[j]]
    NewArray[i][2] = add3

# print(NewArray)

# 求类2的分组情况
NewArray1 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[22:29]
for i in range(Class1, n):
    add1 = 0
    for j in range(0, 7):
        add1 += W4[j] * X[i, G1[j]]
    NewArray1[i - Class1][0] = add1
# 第1组
W5 = W[29:32]
for i in range(Class1, n):
    add2 = 0
    for j in range(0, 3):
        add2 += W5[j] * X[i, G2[j]]
    NewArray1[i - Class1][1] = add2
# 第2组
W6 = W[32:44]
for i in range(Class1, n):
    add3 = 0
    for j in range(0, 12):
        add3 += W6[j] * X[i, G3[j]]
    NewArray1[i - Class1][2] = add3

# print(NewArray1)
# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray, NewArray1))
print(NewArray)
NewArray1 = np.delete(NewArray, T, axis = 1)
df=pd.DataFrame(NewArray1)
sns.pairplot(df)
plt.show()
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



