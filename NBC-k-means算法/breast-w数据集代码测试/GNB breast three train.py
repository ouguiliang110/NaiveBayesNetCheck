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

X = np.loadtxt('[003]breast(0-1).txt')
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
G1 = [4, 6, 8] # 3
G2 = [1, 2, 3, 5, 7, 9] # 6
G3 = [0] # 1

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
train_index1 = [220, 23, 135, 134, 131, 156, 103, 170, 93, 185, 143, 210, 208, 128, 95, 191, 76, 126, 12, 159, 225, 224, 223, 52, 64, 214, 132, 70, 106, 56, 125, 29, 61, 25, 237, 122, 175, 145, 115, 240, 200, 151, 150, 63, 33, 138, 26, 74, 213, 110, 84, 71, 92, 162, 194, 3, 32, 14, 144, 100, 4, 232, 20, 22, 62, 109, 179, 111, 41, 171, 180, 219, 188, 206, 55, 77, 196, 140, 96, 154, 127, 226, 73, 193, 86, 31, 235, 142, 146, 107, 68, 99, 48, 97, 45, 9, 69, 83, 229, 91, 176, 13, 169, 211, 141, 139, 8, 192, 205, 152, 24, 47, 88, 168, 230, 155, 17, 231, 157, 90]
val_index1 = [30, 221, 43, 137, 87, 101, 113, 75, 49, 181, 58, 222, 5, 182, 163, 34, 218, 203, 39, 187, 164, 6, 59, 124, 195, 66, 165, 44, 27, 201, 94, 190, 104, 38, 81, 37, 82, 2, 40, 121, 72, 173, 51, 183, 50, 202, 57, 172]
test_index1 = [0, 1, 7, 10, 11, 15, 16, 18, 19, 21, 28, 35, 36, 42, 46, 53, 54, 60, 65, 67, 78, 79, 80, 85, 89, 98, 102, 105, 108, 112, 114, 116, 117, 118, 119, 120, 123, 129, 130, 133, 136, 147, 148, 149, 153, 158, 160, 161, 166, 167, 174, 177, 178, 184, 186, 189, 197, 198, 199, 204, 207, 209, 212, 215, 216, 217, 227, 228, 233, 234, 236, 238, 239]
train_index2 = [400, 234, 70, 6, 1, 156, 223, 263, 151, 41, 298, 187, 233, 437, 181, 33, 21, 45, 349, 121, 22, 110, 290, 360, 179, 104, 132, 270, 398, 68, 408, 157, 145, 429, 96, 330, 219, 339, 65, 252, 370, 302, 162, 37, 10, 385, 61, 433, 397, 248, 439, 377, 20, 171, 189, 135, 334, 353, 176, 407, 177, 306, 178, 79, 369, 438, 297, 409, 238, 214, 338, 40, 428, 170, 387, 105, 217, 130, 277, 289, 172, 345, 198, 319, 425, 29, 296, 107, 84, 266, 365, 309, 299, 427, 64, 169, 160, 325, 133, 307, 395, 288, 112, 336, 362, 421, 111, 204, 142, 230, 346, 109, 396, 423, 2, 232, 120, 25, 161, 235, 205, 186, 39, 436, 4, 114, 56, 46, 310, 446, 73, 262, 106, 453, 225, 402, 434, 159, 152, 342, 283, 216, 224, 245, 355, 90, 328, 380, 449, 447, 140, 52, 251, 54, 424, 175, 291, 280, 237, 332, 163, 190, 144, 301, 213, 352, 417, 347, 194, 226, 123, 412, 16, 249, 255, 31, 295, 278, 379, 343, 201, 221, 148, 285, 432, 85, 257, 331, 88, 63, 5, 26, 155, 38, 321, 74, 444, 203, 212, 228, 239, 344, 220, 101, 36, 300, 294, 247, 341, 356, 166, 383, 359, 415, 59, 117, 147, 71, 366, 134, 431, 18, 50, 382, 183, 378, 318, 316, 167]
val_index2 = [118, 312, 141, 354, 303, 208, 256, 261, 143, 164, 271, 77, 419, 384, 218, 125, 313, 30, 28, 75, 92, 122, 443, 392, 57, 329, 81, 138, 372, 399, 55, 150, 260, 340, 222, 324, 448, 58, 32, 279, 27, 457, 0, 286, 76, 236, 426, 11, 95, 34, 15, 420, 78, 89, 414, 17, 195, 210, 184, 124, 131, 264, 393, 149, 323, 72, 282, 174, 273, 327, 44, 430, 128, 113, 388, 127, 165, 314, 293, 153, 337, 51, 451, 116, 246, 99, 154, 442, 196, 411, 367]
test_index2 = [3, 7, 8, 9, 12, 13, 14, 19, 23, 24, 35, 42, 43, 47, 48, 49, 53, 60, 62, 66, 67, 69, 80, 82, 83, 86, 87, 91, 93, 94, 97, 98, 100, 102, 103, 108, 115, 119, 126, 129, 136, 137, 139, 146, 158, 168, 173, 180, 182, 185, 188, 191, 192, 193, 197, 199, 200, 202, 206, 207, 209, 211, 215, 227, 229, 231, 240, 241, 242, 243, 244, 250, 253, 254, 258, 259, 265, 267, 268, 269, 272, 274, 275, 276, 281, 284, 287, 292, 304, 305, 308, 311, 315, 317, 320, 322, 326, 333, 335, 348, 350, 351, 357, 358, 361, 363, 364, 368, 371, 373, 374, 375, 376, 381, 386, 389, 390, 391, 394, 401, 403, 404, 405, 406, 410, 413, 416, 418, 422, 435, 440, 441, 445, 450, 452, 454, 455, 456]




#W = getRandom(m * K) * 100
W=[0.49391207194652226, 0.06701034112691243, 0.0939462025306337, 0.5050201157374029, 0.053841110930498086, 0.19040110531440474, 0.048413174512861715, 0.44757214730333106, 0.19764239351490545, 0.11338224860254832, 0.11319690953604361, 0.2903200617023341, 0.14413933290902298, 0.024782959375860718, 0.10101204971163381, 0.211160528170371, 1.1619761002475781, 0.01829979322090902, 0.029028776751019312, 0.14946950222514194, 0.7144617430240148, 0.09038954338773006, 0.11645519957774111, 0.1903889164378073, 0.09484857382417694, 0.012456284876251783, 0.03707189964420932, 0.09014384740422893, 0.011091273505662249, 0.10738551977354738, 0.02773123551044131, 0.11807560560665085, 0.13012008633985095, 0.20100684394215246, 0.06020942069714693, 0.16595062376663972, 0.009663844215810465, 1.0791389920397598, 0.10334597113060418, 0.9777141724517506, 0.07374250618798797, 0.5993960963593532, 0.428606282645425, 0.10607859228112129]


# 求类1的分组情况
NewArray = np.ones((Class1, T + 1))
# 第0组
W1 = W[0:3]
for i in range(0, Class1):
    add1 = 0
    for j in range(0, 3):
        add1 += W1[j] * X[i, G1[j]]
    NewArray[i][0] = add1
# 第1组
W2 = W[3:9]
for i in range(0, Class1):
    add2 = 0
    for j in range(0, 6):
        add2 += W2[j] * X[i, G2[j]]
    NewArray[i][1] = add2
# 第2组
W3 = W[9:10]
for i in range(0, Class1):
    add3 = 0
    for j in range(0, 1):
        add3 += W3[j] * X[i, G3[j]]
    NewArray[i][2] = add3

# print(NewArray)

# 求类2的分组情况
NewArray1 = np.ones((Class2, T + 1)) * 2
# 第0组
W4 = W[10:13]
for i in range(Class1, n):
    add1 = 0
    for j in range(0, 3):
        add1 += W4[j] * X[i, G1[j]]
    NewArray1[i - Class1][0] = add1
# 第1组
W5 = W[13:19]
for i in range(Class1, n):
    add2 = 0
    for j in range(0, 6):
        add2 += W5[j] * X[i, G2[j]]
    NewArray1[i - Class1][1] = add2
# 第2组
W6 = W[19:20]
for i in range(Class1, n):
    add3 = 0
    for j in range(0, 1):
        add3 += W6[j] * X[i, G3[j]]
    NewArray1[i - Class1][2] = add3

# print(NewArray1)
# 合并两个数组，得到真正的合并数据结果
NewArray = np.vstack((NewArray, NewArray1))
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

C1 = clf.predict(trainSet1)
add = sum(C1 == 1)
print("第一类正确数量(总数):", train1)
print(add)
C2 = clf.predict(trainSet2)
add1 = sum(C2 == 2)
print("第二类正确数量(总数)：", train2)
print(add1)

print("accuracy:{:.2%}".format((add + add1) / (train1+train2)))



