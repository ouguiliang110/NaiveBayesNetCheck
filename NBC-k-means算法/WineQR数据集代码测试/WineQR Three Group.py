import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
W = [0.006164268201830558,0.010598321792470052,0.011460240322215397,0.022498170239588076,0.004130568668931144,0.002248332568660909,0.04082372196692318,0.0031782618830056893,0.005902853277869588,0.0008565623792942426,0.02937517176437653,0.0012611932770106539,0.00459123903802768,0.02398194154844883,0.004542242255052139,0.017773323111126973,0.03650060861854068,0.021455480632817707,0.013081344503722133,0.006111595304001227,0.006729752980177407,0.026862836276368998,0.01844973637348151,0.00028778889974891245,0.004524735902732496,0.03151518465690959,0.0021533517770757735,0.009992337602866844,0.0019088164499399451,0.00102783163199139,0.01912569184717601,0.01376070317380448,0.01509497673126374,0.010406827169019132,0.014363343894467778,0.020288214838437957,0.002571968824611091,0.011703163186179723,0.020576197064745623,0.004984535222156354,0.03612476210728883,0.02032733994883157,0.006055974963795745,0.01115796700455957,0.010694817024027655,0.022868617960275582,0.019386954204174692,0.0779557627627168,0.0020794042854982665,0.018908319066047972,0.0004833791814140993,0.01515817975765313,0.024657287105818777,0.011742796464914215,0.02028550147197934,0.014554279210689353,7.283721075395068e-05,0.032561119398626825,0.07043378779981174,0.0015600368834803458,0.00874512122865522,0.00036550499944602714,0.00012042459559307541,0.009791767078533931,0.03367894540517132,0.027365677023173895]
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

# 去掉类标记
NewArray = np.delete(NewArray, T, axis = 1)
df=pd.DataFrame(NewArray)
sns.pairplot(df)
plt.show()
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

# 求各类对应属性的均值和方差
Mean1 = np.mean(trainSet1, axis = 0)
Mean2 = np.mean(trainSet2, axis = 0)
Mean3 = np.mean(trainSet3, axis = 0)
Mean4 = np.mean(trainSet4, axis = 0)
Mean5 = np.mean(trainSet5, axis = 0)
Mean6 = np.mean(trainSet6, axis = 0)

var1 = np.var(trainSet1, axis = 0)
var2 = np.var(trainSet2, axis = 0)
var3 = np.var(trainSet3, axis = 0)
var4 = np.var(trainSet4, axis = 0)
var5 = np.var(trainSet5, axis = 0)
var6 = np.var(trainSet6, axis = 0)
print(var1, var2, var3, var4, var5, var6)

# 先求P(C)
Pro1 = (7 + 1) / (1118 + 6)
Pro2 = (37 + 1) / (1118 + 6)
Pro3 = (477 + 1) / (1118 + 6)
Pro4 = (446 + 1) / (1118 + 6)
Pro5 = (139 + 1) / (1118 + 6)
Pro6 = (12 + 1) / (1118 + 6)
# print(Pro1)
# print(Pro2)

# 统计正确数量和计算准确率
# 计算第一类
add=0
for i in range(0, 7):
    sum=1
    for j in range(0, T):
        sum*=getPro(trainSet1[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, T):
        sum1*=getPro(trainSet1[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, T):
        sum2*=getPro(trainSet1[i][j], Mean3[j], var3[j])
    sum3=  1
    for j in range(0, T):
        sum3*=getPro(trainSet1[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, T):
        sum4*=getPro(trainSet1[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, T):
        sum5*=getPro(trainSet1[i][j], Mean6[j], var6[j])
    if (Pro1 * sum >= Pro2 * sum1) & (Pro1 * sum >= Pro3 * sum2) & (Pro1 * sum >= Pro4 * sum3) & (Pro1 * sum >= Pro5 * sum4) & (Pro1 * sum >= Pro6 * sum5):
        add+=1
    else:
        add+=0
print("第一类正确数量(总数3)：")
print(add)

# 计算第二类
add1=0
for i in range(0, 37):
    sum=1
    for j in range(0, T):
        sum*=getPro(trainSet2[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, T):
        sum1*=getPro(trainSet2[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, T):
        sum2*=getPro(trainSet2[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, T):
        sum3*=getPro(trainSet2[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, T):
        sum4*=getPro(trainSet2[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, T):
        sum5*=getPro(trainSet2[i][j], Mean6[j], var6[j])
    if (Pro2 * sum1 >= Pro1 * sum) & (Pro2 * sum1 >= Pro3 * sum2) & (Pro2 * sum1 >= Pro4 * sum3) & (Pro2 * sum1 >= Pro5 * sum4) & (Pro2 * sum1 >= Pro6 * sum5):
        add1+=1
    else:
        add1+=0
print("第二类正确数量(总数16)：")
print(add1)

# 计算第三类
add2=0
for i in range(0, 477):
    sum=1
    for j in range(0, T):
        sum*=getPro(trainSet3[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, T):
        sum1*=getPro(trainSet3[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, T):
        sum2*=getPro(trainSet3[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, T):
        sum3*=getPro(trainSet3[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, T):
        sum4*=getPro(trainSet3[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, T):
        sum5*=getPro(trainSet3[i][j], Mean6[j], var6[j])
    if (Pro3 * sum2 >= Pro1 * sum) & (Pro3 * sum2 >= Pro2 * sum1) & (Pro3 * sum2 >= Pro4 * sum3) & (Pro3 * sum2 >= Pro5 * sum4) & (Pro3 * sum2 >= Pro6 * sum5):
        add2+=1
    else:
        add2+=0
print("第三类正确数量(总数204)：")
print(add2)

add3=0
for i in range(0, 446):
    sum=1
    for j in range(0, T):
        sum*=getPro(trainSet4[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, T):
        sum1*=getPro(trainSet4[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, T):
        sum2*=getPro(trainSet4[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, T):
        sum3*=getPro(trainSet4[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, T):
        sum4*=getPro(trainSet4[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, T):
        sum5*=getPro(trainSet4[i][j], Mean6[j], var6[j])
    if (Pro4 * sum3 >= Pro1 * sum) & (Pro4 * sum3 >= Pro2 * sum1) & (Pro4 * sum3 >= Pro3 * sum2) & (Pro4 * sum3 >= Pro5 * sum4) & (Pro4 * sum3 >= Pro6 * sum5):
        add3+=1
    else:
        add3+=0
print("第四类正确数量(总数192)：")
print(add3)

add4=0
for i in range(0, 139):
    sum=1
    for j in range(0, T):
        sum*=getPro(trainSet5[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, T):
        sum1*=getPro(trainSet5[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, T):
        sum2*=getPro(trainSet5[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, T):
        sum3*=getPro(trainSet5[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, T):
        sum4*=getPro(trainSet5[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, T):
        sum5*=getPro(trainSet5[i][j], Mean6[j], var6[j])
    if (Pro5 * sum4 >= Pro1 * sum) & (Pro5 * sum4 >= Pro2 * sum1) & (Pro5 * sum4 >= Pro3 * sum2) & (Pro5 * sum4 >= Pro4 * sum3) & (Pro5 * sum4 >= Pro6 * sum5):
        add4+=1
    else:
        add4+=0
print("第五类正确数量(总数60)：")
print(add4)

add5=0
for i in range(0, 12):
    sum=1
    for j in range(0, T):
        sum*=getPro(trainSet6[i][j], Mean1[j], var1[j])
    sum1=1
    for j in range(0, T):
        sum1*=getPro(trainSet6[i][j], Mean2[j], var2[j])
    sum2=1
    for j in range(0, T):
        sum2*=getPro(trainSet6[i][j], Mean3[j], var3[j])
    sum3=1
    for j in range(0, T):
        sum3*=getPro(trainSet6[i][j], Mean4[j], var4[j])
    sum4=1
    for j in range(0, T):
        sum4*=getPro(trainSet6[i][j], Mean5[j], var5[j])
    sum5=1
    for j in range(0, T):
        sum5*=getPro(trainSet6[i][j], Mean6[j], var6[j])
    if (Pro6 * sum5 >= Pro1 * sum) & (Pro6 * sum5 >= Pro2 * sum1) & (Pro6 * sum5 >= Pro3 * sum2) & (Pro6 * sum5 >= Pro4 * sum3) & (Pro6 * sum5 >= Pro5 * sum4):
        add5+=1
    else:
        add5+=0
print("第六类正确数量(总数6)：")
print(add5)

# 准确率
print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5) / 1118))
