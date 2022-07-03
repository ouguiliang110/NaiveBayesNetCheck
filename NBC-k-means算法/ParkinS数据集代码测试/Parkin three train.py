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
train_index1 = [134, 57, 40, 102, 2, 112, 117, 73, 54, 66, 121, 92, 42, 106, 123, 97, 37, 77, 32, 14, 26, 11, 113, 122, 84, 146, 51, 94, 127, 19, 132, 99, 67, 21, 135, 64, 119, 143, 31, 0, 4, 142, 39, 86, 125, 65, 115, 61, 85, 80, 3, 103, 43, 108, 36, 90, 16, 124, 137, 8, 30, 23, 107, 52, 24, 139, 46, 72, 60, 15, 17, 70, 140]
val_index1 = [27, 35, 9, 83, 88, 95, 114, 34, 62, 29, 116, 129, 44, 128, 75, 120, 89, 71, 7, 20, 74, 81, 10, 69, 58, 145, 144, 50, 49]
test_index1 = [1, 5, 6, 12, 13, 18, 22, 25, 28, 33, 38, 41, 45, 47, 48, 53, 55, 56, 59, 63, 68, 76, 78, 79, 82, 87, 91, 93, 96, 98, 100, 101, 104, 105, 109, 110, 111, 118, 126, 130, 131, 133, 136, 138, 141]
train_index2 = [6, 27, 1, 26, 46, 18, 16, 2, 45, 5, 12, 22, 47, 43, 23, 3, 8, 4, 35, 0, 14, 36, 24, 37]
val_index2 = [29, 10, 30, 20, 42, 15, 33, 44, 11]
test_index2 = [7, 9, 13, 17, 19, 21, 25, 28, 31, 32, 34, 38, 39, 40, 41]




#W = getRandom(m * K) * 100
W=[0.5454489129017815, 0.24707864613438812, 0.006013800495820629, 0.018988361087771997, 0.16994821614403546, 0.0493483707327571, 0.24551437958619715, 0.8798980004826384, 0.16222514511315902, 0.24145494870116435, 0.17654756771851274, 0.024967704937601445, 0.04652661481603247, 1.2391339300956001, 0.7456391992514364, 0.11527819892745536, 0.054380049171428205, 0.1502985170644243, 0.26026981224896, 0.04735310327304052, 0.16257447694966898, 0.29857496063255184, 0.4210889507829293, 0.1403014764621924, 0.7026627731614267, 0.14941058393432782, 0.18931876591743968, 0.1591244010569938, 0.5235159552541268, 0.043141833630597884, 0.6499659875204302, 0.040772933190168954, 0.01972287842208626, 0.0758090305838932, 0.0929557296905564, 0.12305424285324147, 0.13325442106182075, 0.04131269490973012, 0.08701792843364982, 0.08882489843770915, 0.26933582440790327, 0.00981110850824577, 0.01135680847043466, 0.14077785684366745]

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
# print(NewArray)

# 随机抽取样本训练集和测试集样本

X1 = NewArray[0:Class1, :]
X2 = NewArray[Class1:Class1 + Class2, :]

Data1 = X1[train_index1, :]
Data2 = X2[train_index2, :]

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

# 先求P(C)
Pro1 = (train1 + 1) / (train1 + train2 + 1)
Pro2 = (train2 + 1) / (train1 + train2 + 1)

add = 0
for i in range(0, train1):
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
print("第一类正确数量(总数):", val1)
print(add)
add1 = 0
for i in range(0,train2):
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
print("第二类正确数量(总数)：", val2)
print(add1)
print("accuracy:{:.2%}".format((add + add1) / (train2+train1)))
