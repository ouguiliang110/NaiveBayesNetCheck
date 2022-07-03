import numpy as np
import math


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
#W = [0.002325854739685328,0.00021838564295841855,0.02985475702298483,0.0518446559495882,0.02012641052796597,0.007409340127033962,0.022201496099853314,0.03012129144087563,0.01454428364624264,0.0022398234406778446,0.0056657923615670465,0.016659890799380614,0.018056756520631708,0.008485812767169082,0.004710364609461818,0.04497221844227798,0.0010515744333933709,0.006695038521268798,0.003091404316910227,0.01263611254003536,0.0024637239395477683,0.03245251968353364,0.023955118985052894,0.012931957174835517,0.0190429502990245,0.020339584782561107,0.008406241156150835,0.0029795606750892306,0.013441945815733079,0.02444044546992782,0.00791015657437148,0.027059144997403157,0.012588392312071288,0.004520098257657641,0.007477778682764242,0.007230115147414573,0.06520604327470197,0.02568570289691831,0.012741296319852224,0.0015073137765023452,0.00044915637933533407,0.009847473588276666,0.047566064742072596,0.011857758095603136,0.017267143325511502,2.749276313694238e-05,0.0016352399963503778,0.030117271850691442,0.011028449881476922,0.0007460190785953783,0.03637859979412315,0.005243656817445851,0.00450305973007181,0.013703047190657899,0.004713807581306524,0.004041915631752294,0.0006442822142901059,0.004464995132599896,0.06863073150656754,0.0013205245194687228,0.00599597348721585,0.036496507532145964,0.008050349980234649,0.01676902413705888,0.0005111760322809633,0.022698924840653632]
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
