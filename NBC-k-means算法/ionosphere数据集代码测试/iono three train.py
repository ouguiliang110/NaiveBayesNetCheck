import numpy as np
import math


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    a=1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro=1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
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
G1 = [6, 13, 14, 24, 30]  # 5
G2 = [5, 7, 9, 17, 21, 25, 31]# 7
G3 = [0, 1, 2, 3, 4, 8, 10, 11, 12, 15, 16, 18, 19, 20, 22, 23, 26, 27, 28, 29]# 20
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
train_index1 = [103, 35, 117, 65, 100, 5, 87, 3, 20, 58, 10, 77, 13, 122, 114, 53, 17, 52, 38, 69, 83, 120, 56, 14, 96, 25, 6, 21, 51, 105, 125, 112, 104, 107, 86, 12, 95, 68, 79, 40, 28, 46, 27, 110, 63, 97, 24, 1, 15, 9, 91, 41, 98, 54, 62, 36, 101, 123, 93, 43, 49, 99, 0]
val_index1 = [85, 78, 115, 30, 75, 106, 76, 109, 72, 32, 42, 60, 34, 33, 48, 118, 73, 45, 74, 18, 88, 90, 7, 22, 29]
test_index1 = [2, 4, 8, 11, 16, 19, 23, 26, 31, 37, 39, 44, 47, 50, 55, 57, 59, 61, 64, 66, 67, 70, 71, 80, 81, 82, 84, 89, 92, 94, 102, 108, 111, 113, 116, 119, 121, 124]
train_index2 = [142, 204, 180, 58, 190, 217, 224, 178, 78, 128, 211, 116, 206, 57, 2, 12, 174, 56, 153, 64, 97, 146, 37, 55, 100, 172, 208, 191, 89, 182, 115, 158, 85, 107, 84, 40, 162, 94, 83, 109, 139, 17, 32, 186, 18, 215, 145, 188, 147, 68, 67, 106, 168, 43, 74, 126, 133, 87, 49, 16, 8, 203, 134, 144, 160, 72, 31, 149, 23, 9, 127, 92, 200, 120, 7, 0, 189, 135, 99, 169, 95, 10, 81, 137, 101, 176, 14, 148, 53, 117, 143, 88, 63, 140, 219, 177, 121, 20, 194, 15, 207, 114, 61, 28, 50, 52, 103, 104, 222, 170, 51, 69]
val_index2 = [75, 80, 111, 155, 73, 19, 29, 45, 157, 193, 156, 118, 130, 47, 13, 102, 4, 113, 93, 202, 54, 129, 36, 212, 59, 125, 150, 163, 175, 66, 35, 21, 112, 171, 152, 218, 119, 41, 71, 3, 205, 11, 33, 216, 213]
test_index2 = [1, 5, 6, 22, 24, 25, 26, 27, 30, 34, 38, 39, 42, 44, 46, 48, 60, 62, 65, 70, 76, 77, 79, 82, 86, 90, 91, 96, 98, 105, 108, 110, 122, 123, 124, 131, 132, 136, 138, 141, 151, 154, 159, 161, 164, 165, 166, 167, 173, 179, 181, 183, 184, 185, 187, 192, 195, 196, 197, 198, 199, 201, 209, 210, 214, 220, 221, 223]
W = getRandom(m * K) * 100
W=[0.41297339398166916, 0.5701531726297927, 1.1792191035540536, 0.5334774336295915, 0.43012273264027023, 0.38111590018231545, 0.3122600053614788, 0.16372729259847285, 1.191077239532864, 0.2149164055836515, 0.11111336873959868, 0.8659226989555975, 0.535797039845662, 0.3275927137710348, 0.8623558504801607, 0.9427575028017887, 0.6931503939134811, 0.2291636962134964, 0.7207712577176929, 0.12384595642260052, 0.3615775946960843, 0.68515628397579, 0.6303557410836815, 0.7309002365275814, 0.4177875007863854, 2.6213487946392444, 1.279252520785225, 0.588950389009153, 1.2711346699267718, 0.7627286067667363, 0.16911923664757122, 0.7772063823697266, 0.2377584891978458, 0.5029919758394469, 0.11198061283692254, 0.2974992118554844, 0.08885105146825029, 0.2666338689685257, 0.6254866701722712, 0.4012217229640124, 0.09363781079195568, 0.5616868519134779, 0.7060813520339563, 1.072873870319829, 0.2329181413338695, 0.26910894177676203, 0.39006556486219757, 0.3131850635478031, 0.8489183714914914, 0.6240687003452159, 0.6773325107881141, 1.0333162295300249, 0.8009351454091566, 0.8044731532398386, 0.5336476454703578, 0.8561458802214159, 0.5460680989283474, 0.5173577307822422, 0.9121435703080145, 1.520824186429527, 0.4141177493215137, 0.8267705341794207, 0.49643344778078746, 0.17534462279854532]



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
print(NewArray)
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
