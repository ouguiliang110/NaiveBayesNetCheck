import numpy as np
import random
import pandas as pd


# 主要求E的值
def getE(l, j):
    E = 0
    for i in range(0, n):
        E = E + U[i][l] * (X[i][j] - Z[l][j]) ** 2
    E = E + n1
    return E


# 求GY
def getGY(j, l):
    add2 = 0
    for t in range(0, T):
        add2 = add2 + G[j][t] * (Y[l][t] ** 2)
    return add2


# 求GYT
def getGYV(j, l):
    add1 = 0
    for t in range(0, T):
        add1 = add1 + G[j][t] * (Y[l][t] ** 2) * V[l][t]
    return add1


# 求其中F

def getF(j, t):
    add8 = 0
    for l in range(0, K):
        add8 += (Y[l, t] ** 2) * (W[l, j] - V[l, t]) ** 2
    return add8


# 求其中H

def getH(l, t):
    add8 = 0
    for j in range(0, m):
        add8 += G[j, t] * (W[l, j] - V[l, t]) ** 2
    return add8 + n2


X = np.loadtxt('[028]wineQR(0-1).txt')
# 主要的参数设置
n1 = 0.0001
n2 = 0.0001
n3 = 0.00001
N = 50
b = 1  # β值大小
T = 3  # 组数量大小
m = 11  # 属性数量
n = 1599  # 样本数目
K = 6  # 类标记数量
Z = np.zeros((K, m))
U = np.zeros((n, K))
#print(X.shape)
# print(U)
# print(X)
p = 0
p1 = 0
p2 = 0
p3 = 0
p4 = 0
p5 = 0
p6 = 0
# 初始化可得出六个类数量10,53,681,638,199,18
for i in X:
    if i[11] == 1:
        U[p][0] = 1
        p1 = p1 + 1
    elif i[11] == 2:
        U[p][1] = 1
        p2 = p2 + 1
    elif i[11] == 3:
        U[p][2] = 1
        p3 = p3 + 1
    elif i[11] == 4:
        U[p][3] = 1
        p4 = p4 + 1
    elif i[11] == 5:
        U[p][4] = 1
        p5 = p5 + 1
    elif i[11] == 6:
        U[p][5] = 1
        p6 = p6 + 1
X = np.delete(X, 11, axis = 1)

X1 = X[0:10, :]
X2 = X[10:63, :]
X3 = X[63:744, :]
X4 = X[744:1382, :]
X5 = X[1382:1581, :]
X6 = X[1581:1599, :]

# 初始化Z矩阵
Z[0] = np.mean(X1, axis = 0)
Z[1] = np.mean(X2, axis = 0)
Z[2] = np.mean(X3, axis = 0)
Z[3] = np.mean(X4, axis = 0)
Z[4] = np.mean(X5, axis = 0)
Z[5] = np.mean(X6, axis = 0)

# 初始化类属性W，属性分组矩阵y，类组矩阵V为1
W = np.ones((K, m))
V = np.ones((K, T))
Y = np.ones((K, T))

# 初始化G
G = np.zeros((m, T))
G[:, 0] = 1
# print(G)
# 初始化s,Q
'''标记循环次数的主要参数'''
s = 0
'''目标函数的大小'''
Q = 0
'''循环次数'''
goNum = 0
'''
for i in range(0, K):
    print(i)
'''
while True:
    # 求W 其中最难最复杂的内容
    for l in range(0, K):
        for j in range(0, m):
            # 求h1的值
            add1 = getGYV(j, l)
            h1 = b * add1
            # 求h2的值
            add2 = getGY(j, l)
            add2 = b * add2
            E1 = getE(l, j)
            h2 = add2 + E1
            # 求h3的值
            add4 = 0
            for h in range(0, m):
                num1 = getGY(h, l)
                num1 = num1 * b
                E2 = getE(l, h)
                num2 = getGYV(h, l)
                num2 = num2 * b
                add4 = add4 + (1 / (num1 + E2)) * num2
            h3 = add4 - m
            # 求h4的值
            add5 = 0
            for h in range(0, m):
                num3 = b * getGY(h, l)
                E3 = getE(l, h)
                add5 = add5 + 1 / (num3 + E3)
            h4 = (b * getGY(j, l) + getE(l, j)) * add5
            W[l][j] = h1 / h2 - h3 / h4
    # 主要条件判断更新V矩阵
    if s == 0:
        # 初始化V1
        for t in range(0, T):
            ranNum = random.randint(0, m - 1)
            V[:, t] = W[:, ranNum]
    else:
        add6 = 0
        add7 = 0
        for l in range(0, K):
            for t in range(0, T):
                for j in range(0, m):
                    add6 = add6 + G[j, t] * W[l, j]
                    add7 = add7 + G[j, t]
                V[l, t] = add6 / add7
    # 更新G矩阵
    for j in range(0, m):
        for t in range(0, T):
            Ft = getF(j, t)
            flag = 1
            for s in range(0, T):
                if Ft > getF(j, s):
                    flag = 0
            G[j, t] = flag

    # 更新Y矩阵
    for l in range(0, K):
        for t in range(0, T):
            add9 = 0
            for s in range(0, K):
                add9 += getH(l, t) / getH(s, t)
            Y[l, t] = 2 / add9
    # 主要目标函数进行求解
    s = s + 1
    Qnum1 = 0;
    for l in range(0, K):
        for i in range(0, n):
            add10 = 0
            for j in range(0, m):
                add10 += (W[l, j] ** 2) * ((X[i, j] - Z[l, j]) ** 2)
            Qnum1 += U[i, l] * add10
    Qnum2 = 0
    for l in range(0, K):
        for j in range(0, m):
            Qnum2 += W[l, j] ** 2
    Qnum2 = Qnum2 * n1
    Qnum3 = 0
    for t in range(0, T):
        for j in range(0, m):
            add11 = 0
            for l in range(0, K):
                add11 += (Y[l, t] ** 2) * ((W[l, j] - V[l, t]) ** 2)
            Qnum3 += G[j, t] * add11
    Qnum3 = Qnum3 * b
    Qnum4 = 0
    for l in range(0, K):
        for t in range(0, T):
            Qnum4 += Y[l, t] ** 2
    Qnum4 = n2 * Qnum4 * b
    Q1 = Qnum1 + Qnum2 + Qnum3 + Qnum4
    if abs(Q1 - Q) < n3 or s >= N:
        break
    Q = Q1
    goNum += 1
    print(goNum)
    print(np.sum(G, axis = 0))
print(W)
# print(np.sum(W,axis = 1))
print("---------------------")
print(G)
print(np.sum(G, axis = 0))
print("----------------------")
for t in range(0, T):
    print("第" + repr(t) + "组")
    list = []
    for j in range(0, m):
        if G[j, t] == 1:
            list.append(j)
        else:
            continue
    print(list)
    print('\n')
print("---------------------")
# print(V)