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
#求其中F

def getF(j, t):
    add8 = 0
    for l in range(0, K):
        add8 += (Y[l, t] ** 2) * (W[l, j] - V[l, t]) ** 2
    return add8
#求其中H

def getH(l, t):
    add8 = 0
    for j in range(0, m):
        add8 += G[j, t] * (W[l, j] - V[l, t]) ** 2
    return add8 + n2


X = np.loadtxt('[017]magic(0-1).txt')
# 主要的参数设置
n1 = 0.0001
n2 = 0.0001
n3 = 0.000001
N = 100
b = 1  # β值大小
T = 5  # 组数量大小
m = 10  # 属性数量
n = 1902  # 样本数目 其中第一类1219，第二类683
K = 2  # 类标记数量
Z = np.zeros((K, m))
U = np.zeros((n, K))
#print(X.shape)
# print(U)
# print(X)
p = 0
p1 = 0
# 创建空数组，为了按顺序合并1和2分类
'''
newarray1 = np.empty(shape = [0, 45])
newarray2 = np.empty(shape = [0, 45])
for i in X:
    if i[44] == 1:
        newarray1 = np.vstack((newarray1, i))
    elif i[44]==2:
        newarray2 = np.vstack((newarray2, i))
newarray = np.vstack((newarray1, newarray2))
X=newarray
'''
#print(newarray)
#统计各类数量
# 初始化
for i in X:
    if i[10] == 1:
        U[p][0] = 1
        p1 = p1 + 1
    elif i[10] == 2:
        U[p][1] = 1
    p = p + 1
#print(p1)
#print(p)
X = np.delete(X, 10, axis = 1)

# 取得除去类列外的剩余值，这里主要求Z
# print(X)
X1 = X[0:1219, :]
X2 = X[1219:, :]
# 初始化Z矩阵
# 初始化Z矩阵
Z[0] = np.mean(X1, axis = 0)
Z[1] = np.mean(X2, axis = 0)

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
# 主要的循环过程
# 由算法可知类标记已经确定，则Z和U无需更新，只需要更新W
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
    Qnum1 = 0
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
#print(np.sum(W,axis = 1))
print("---------------------")
print(G)
print(np.sum(G,axis = 0))
print("----------------------")
for t in range(0,T):
    print("第" + repr(t) + "组")
    list = []
    for j in range(0,m):
           if G[j,t]==1:
               list.append(j)
           else:
               continue
    print(list)
    print('\n')
print("---------------------")
#print(V)