import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


# 连续型数据分类用正态分布公式
def getPro(theData, mean, var):
    a=1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro=1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro


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

X = np.loadtxt('[013]segment(0-1).txt')
X=X[:,np.append(np.arange(9,18),[X.shape[1]-1])]
NewArray = np.delete(X, X.shape[1]-1, axis = 1)
df=pd.DataFrame(NewArray)
sns.heatmap(df.corr(),annot = True)
plt.show()
# 其中有97
m = X.shape[1] - 1  # 属性数量
n = X.shape[0]  # 样本数目
K = 7  # 类标记数量

# 去掉类标记
Class1 = 0
Class2 = 0
Class3 = 0
Class4 = 0
Class5 = 0
Class6 = 0
Class7 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1
    elif i[m] == 3:
        Class3 = Class3 + 1
    elif i[m] == 4:
        Class4 = Class4 + 1
    elif i[m] == 5:
        Class5 = Class5 + 1
    elif i[m] == 6:
        Class6 = Class6 + 1
    elif i[m] == 7:
        Class7 = Class7 + 1

train1 = int(Class1 * 0.7)
test1 = Class1 - train1

train2 = int(Class2 * 0.7)
test2 = Class2 - train2

train3 = int(Class3 * 0.7)
test3 = Class3 - train3

train4 = int(Class4 * 0.7)
test4 = Class4 - train4

train5 = int(Class5 * 0.7)
test5 = Class5 - train5

train6 = int(Class6 * 0.7)
test6 = Class6 - train6

train7 = int(Class7 * 0.7)
test7 = Class7 - train7

num2 = Class1 + Class2
num3 = Class1 + Class2 + Class3
num4 = num3 + Class4
num5 = num4 + Class5
num6 = num5 + Class6
num7 = num6 + Class7

# print(X)
X1 = X[0:Class1, :]
X2 = X[Class1:num2, :]
X3 = X[num2:num3, :]
X4 = X[num3:num4, :]
# print(X4)
X5 = X[num4:num5, :]
# print(X5)
X6 = X[num5:num6, :]
X7 = X[num6:num7, :]

acc = []
for i in range(0, 20):
    # 随机抽取样本训练集和测试集样本
    idx = np.random.choice(np.arange(Class1), size = train1, replace = False)
    train_index1 = np.array(idx)
    test_index1 = np.delete(np.arange(Class1), train_index1)

    idx1 = np.random.choice(np.arange(Class2), size = train2, replace = False)
    train_index2 = np.array(idx1)
    test_index2 = np.delete(np.arange(Class2), train_index2)

    idx2 = np.random.choice(np.arange(Class3), size = train3, replace = False)
    train_index3 = np.array(idx2)
    test_index3 = np.delete(np.arange(Class3), train_index3)

    idx3 = np.random.choice(np.arange(Class4), size = train4, replace = False)
    train_index4 = np.array(idx3)
    test_index4 = np.delete(np.arange(Class4), train_index4)

    idx4 = np.random.choice(np.arange(Class5), size = train5, replace = False)
    train_index5 = np.array(idx4)
    test_index5 = np.delete(np.arange(Class5), train_index5)

    idx5 = np.random.choice(np.arange(Class6), size = train6, replace = False)
    train_index6 = np.array(idx5)
    test_index6 = np.delete(np.arange(Class6), train_index6)

    idx6 = np.random.choice(np.arange(Class7), size = train7, replace = False)
    train_index7 = np.array(idx6)
    test_index7 = np.delete(np.arange(Class7), train_index7)

    Data1 = X1[train_index1, :]
    Data2 = X2[train_index2, :]
    Data3 = X3[train_index3, :]
    Data4 = X4[train_index4, :]
    Data5 = X5[train_index5, :]
    Data6 = X6[train_index6, :]
    Data7 = X7[train_index7, :]

    testSet1 = np.delete(X1[test_index1, :], m, axis = 1)
    testSet2 = np.delete(X2[test_index2, :], m, axis = 1)
    testSet3 = np.delete(X3[test_index3, :], m, axis = 1)
    testSet4 = np.delete(X4[test_index4, :], m, axis = 1)
    testSet5 = np.delete(X5[test_index5, :], m, axis = 1)
    testSet6 = np.delete(X6[test_index6, :], m, axis = 1)
    testSet7 = np.delete(X7[test_index7, :], m, axis = 1)


    # 求各类对应属性的均值和方差
    Mean1 = np.mean(np.delete(Data1, m, axis = 1), axis = 0)
    Mean2 = np.mean(np.delete(Data2, m, axis = 1), axis = 0)
    Mean3 = np.mean(np.delete(Data3, m, axis = 1), axis = 0)
    Mean4 = np.mean(np.delete(Data4, m, axis = 1), axis = 0)
    Mean5 = np.mean(np.delete(Data5, m, axis = 1), axis = 0)
    Mean6 = np.mean(np.delete(Data6, m, axis = 1), axis = 0)
    Mean7 = np.mean(np.delete(Data7, m, axis = 1), axis = 0)

    var1 = np.mean(np.delete(Data1, m, axis = 1), axis = 0)
    var2 = np.mean(np.delete(Data2, m, axis = 1), axis = 0)
    var3 = np.mean(np.delete(Data3, m, axis = 1), axis = 0)
    var4 = np.mean(np.delete(Data4, m, axis = 1), axis = 0)
    var5 = np.mean(np.delete(Data5, m, axis = 1), axis = 0)
    var6 = np.mean(np.delete(Data6, m, axis = 1), axis = 0)
    var7 = np.mean(np.delete(Data7, m, axis = 1), axis = 0)
    alltrain = train1 + train2 + train3 + train4 + train5 + train6 + train7
    # 先求P(C)
    Pro1 = (train1 + 1) / (alltrain + 7)
    Pro2 = (train2 + 1) / (alltrain + 7)
    Pro3 = (train3 + 1) / (alltrain + 7)
    Pro4 = (train4 + 1) / (alltrain + 7)
    Pro5 = (train5 + 1) / (alltrain + 7)
    Pro6 = (train6 + 1) / (alltrain + 7)
    Pro7 = (train7 + 1) / (alltrain + 7)

    add = 0
    for i in range(0, test1):
        sum = 1
        for j in range(0, m):
            sum *= getPro(testSet1[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(testSet1[i][j], Mean2[j], var2[j])
        sum2 = 1
        for j in range(0, m):
            sum2 *= getPro(testSet1[i][j], Mean3[j], var3[j])
        sum3 = 1
        for j in range(0, m):
            sum3 *= getPro(testSet1[i][j], Mean4[j], var4[j])
        sum4 = 1
        for j in range(0, m):
            sum4 *= getPro(testSet1[i][j], Mean5[j], var5[j])
        sum5 = 1
        for j in range(0, m):
            sum5 *= getPro(testSet1[i][j], Mean6[j], var6[j])
        sum6 = 1
        for j in range(0, m):
            sum6 *= getPro(testSet1[i][j], Mean7[j], var7[j])
        if (Pro1 * sum >= Pro2 * sum1) & (Pro1 * sum >= Pro3 * sum2) & (Pro1 * sum >= Pro4 * sum3) & (
                Pro1 * sum >= Pro5 * sum4) & (Pro1 * sum >= Pro6 * sum5) & (Pro1 * sum >= Pro7 * sum6):
            add += 1
        else:
            add += 0
    print("第一类正确数量(总数27)：")
    print(add)
    add1 = 0
    for i in range(0, test2):
        sum = 1
        for j in range(0, m):
            sum *= getPro(testSet2[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(testSet2[i][j], Mean2[j], var2[j])
        sum2 = 1
        for j in range(0, m):
            sum2 *= getPro(testSet2[i][j], Mean3[j], var3[j])
        sum3 = 1
        for j in range(0, m):
            sum3 *= getPro(testSet2[i][j], Mean4[j], var4[j])
        sum4 = 1
        for j in range(0, m):
            sum4 *= getPro(testSet2[i][j], Mean5[j], var5[j])
        sum5 = 1
        for j in range(0, m):
            sum5 *= getPro(testSet2[i][j], Mean6[j], var6[j])
        sum6 = 1
        for j in range(0, m):
            sum6 *= getPro(testSet2[i][j], Mean7[j], var7[j])
        if (Pro2 * sum1 >= Pro1 * sum) & (Pro2 * sum1 >= Pro3 * sum2) & (Pro2 * sum1 >= Pro4 * sum3) & (
                Pro2 * sum1 >= Pro5 * sum4) & (Pro2 * sum1 >= Pro6 * sum5) & (Pro2 * sum1 >= Pro7 * sum6):
            add1 += 1
        else:
            add1 += 0
    print("第二类正确数量(总数34)：")
    print(add1)

    # 计算第三类
    add2 = 0
    for i in range(0, test3):
        sum = 1
        for j in range(0, m):
            sum *= getPro(testSet3[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(testSet3[i][j], Mean2[j], var2[j])
        sum2 = 1
        for j in range(0, m):
            sum2 *= getPro(testSet3[i][j], Mean3[j], var3[j])
        sum3 = 1
        for j in range(0, m):
            sum3 *= getPro(testSet3[i][j], Mean4[j], var4[j])
        sum4 = 1
        for j in range(0, m):
            sum4 *= getPro(testSet3[i][j], Mean5[j], var5[j])
        sum5 = 1
        for j in range(0, m):
            sum5 *= getPro(testSet3[i][j], Mean6[j], var6[j])
        sum6 = 1
        for j in range(0, m):
            sum6 *= getPro(testSet3[i][j], Mean7[j], var7[j])
        if (Pro3 * sum2 >= Pro1 * sum) & (Pro3 * sum2 >= Pro2 * sum1) & (Pro3 * sum2 >= Pro4 * sum3) & (
                Pro3 * sum2 >= Pro5 * sum4) & (Pro3 * sum2 >= Pro6 * sum5) & (Pro3 * sum2 >= Pro7 * sum6):
            add2 += 1
        else:
            add2 += 0
    print("第三类正确数量(总数204)：")
    print(add2)

    add3 = 0
    for i in range(0, test4):
        sum = 1
        for j in range(0, m):
            sum *= getPro(testSet4[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(testSet4[i][j], Mean2[j], var2[j])
        sum2 = 1
        for j in range(0, m):
            sum2 *= getPro(testSet4[i][j], Mean3[j], var3[j])
        sum3 = 1
        for j in range(0, m):
            sum3 *= getPro(testSet4[i][j], Mean4[j], var4[j])
        sum4 = 1
        for j in range(0, m):
            sum4 *= getPro(testSet4[i][j], Mean5[j], var5[j])
        sum5 = 1
        for j in range(0, m):
            sum5 *= getPro(testSet4[i][j], Mean6[j], var6[j])
        sum6 = 1
        for j in range(0, m):
            sum6 *= getPro(testSet4[i][j], Mean7[j], var7[j])
        if (Pro4 * sum3 >= Pro1 * sum) & (Pro4 * sum3 >= Pro2 * sum1) & (Pro4 * sum3 >= Pro3 * sum2) & (
                Pro4 * sum3 >= Pro5 * sum4) & (Pro4 * sum3 >= Pro6 * sum5) & (Pro4 * sum3 >= Pro7 * sum6):
            add3 += 1
        else:
            add3 += 0
    print("第四类正确数量(总数192)：")
    print(add3)

    add4 = 0
    for i in range(0, test5):
        sum = 1
        for j in range(0, m):
            sum *= getPro(testSet5[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(testSet5[i][j], Mean2[j], var2[j])
        sum2 = 1
        for j in range(0, m):
            sum2 *= getPro(testSet5[i][j], Mean3[j], var3[j])
        sum3 = 1
        for j in range(0, m):
            sum3 *= getPro(testSet5[i][j], Mean4[j], var4[j])
        sum4 = 1
        for j in range(0, m):
            sum4 *= getPro(testSet5[i][j], Mean5[j], var5[j])
        sum5 = 1
        for j in range(0, m):
            sum5 *= getPro(testSet5[i][j], Mean6[j], var6[j])
        sum6 = 1
        for j in range(0, m):
            sum6 *= getPro(testSet5[i][j], Mean7[j], var7[j])
        if (Pro5 * sum4 >= Pro1 * sum) & (Pro5 * sum4 >= Pro2 * sum1) & (Pro5 * sum4 >= Pro3 * sum2) & (
                Pro5 * sum4 >= Pro4 * sum3) & (Pro5 * sum4 >= Pro6 * sum5) & (Pro5 * sum4 >= Pro7 * sum6):
            add4 += 1
        else:
            add4 += 0
    print("第五类正确数量(总数60)：")
    print(add4)

    add5 = 0
    for i in range(0, test6):
        sum = 1
        for j in range(0, m):
            sum *= getPro(testSet6[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(testSet6[i][j], Mean2[j], var2[j])
        sum2 = 1
        for j in range(0, m):
            sum2 *= getPro(testSet6[i][j], Mean3[j], var3[j])
        sum3 = 1
        for j in range(0, m):
            sum3 *= getPro(testSet6[i][j], Mean4[j], var4[j])
        sum4 = 1
        for j in range(0, m):
            sum4 *= getPro(testSet6[i][j], Mean5[j], var5[j])
        sum5 = 1
        for j in range(0, m):
            sum5 *= getPro(testSet6[i][j], Mean6[j], var6[j])
        sum6 = 1
        for j in range(0, m):
            sum6 *= getPro(testSet6[i][j], Mean7[j], var7[j])
        if (Pro6 * sum5 >= Pro1 * sum) & (Pro6 * sum5 >= Pro2 * sum1) & (Pro6 * sum5 >= Pro3 * sum2) & (
                Pro6 * sum5 >= Pro4 * sum3) & (Pro6 * sum5 >= Pro5 * sum4) & (Pro6 * sum5 >= Pro7 * sum6):
            add5 += 1
        else:
            add5 += 0
    print("第六类正确数量(总数6)：")
    print(add5)
    add6 = 0
    for i in range(0, test7):
        sum = 1
        for j in range(0, m):
            sum *= getPro(testSet7[i][j], Mean1[j], var1[j])
        sum1 = 1
        for j in range(0, m):
            sum1 *= getPro(testSet7[i][j], Mean2[j], var2[j])
        sum2 = 1
        for j in range(0, m):
            sum2 *= getPro(testSet7[i][j], Mean3[j], var3[j])
        sum3 = 1
        for j in range(0, m):
            sum3 *= getPro(testSet7[i][j], Mean4[j], var4[j])
        sum4 = 1
        for j in range(0, m):
            sum4 *= getPro(testSet7[i][j], Mean5[j], var5[j])
        sum5 = 1
        for j in range(0, m):
            sum5 *= getPro(testSet7[i][j], Mean6[j], var6[j])
        sum6 = 1
        for j in range(0, m):
            sum6 *= getPro(testSet7[i][j], Mean7[j], var7[j])
        if (Pro7 * sum6 >= Pro1 * sum) & (Pro7 * sum6 >= Pro2 * sum1) & (Pro7 * sum6 >= Pro3 * sum2) & (
                Pro7 * sum6 >= Pro4 * sum3) & (Pro7 * sum6 >= Pro5 * sum4) & (Pro7 * sum6 >= Pro6 * sum5):
            add6 += 1
        else:
            add6 += 0
    print("第六类正确数量(总数6)：")
    print(add6)

    acc.append((add + add1 + add2 + add3 + add4 + add5 + add6) / (
                test1 + test2 + test3 + test4+test5 + test6 + test7))
    print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5 + add6) / (
                test1 + test2 + test3 + test4+test5 + test6 + test7)))
arr_mean = np.mean(acc)
# 求方差
arr_var = np.var(acc)
arr_std = np.std(acc, ddof = 1)
print(arr_mean, arr_var, arr_std)
