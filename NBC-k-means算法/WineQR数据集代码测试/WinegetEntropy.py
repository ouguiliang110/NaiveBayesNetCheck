import numpy as np
import math
import random
import pandas as pd


def getE(getX):
    N = getX.shape[0]
    Ed = []
    for d in range(0, 11):
        E = 0
        for n in range(0, N):
            F = 0
            for m in range(0, N):
                if n != m:
                    F += 1 / (math.sqrt(2 * math.pi) * h) * math.exp((-1 / 2) * ((getX[n, d] - getX[m, d]) / h) ** 2)
                else:
                    continue
            F = 1 / (N - 1) * F
            E += math.log(F, math.e)
        E = E * (-1 / N)
        Ed.append(E)
    return Ed


def getRound(E):
    R = []
    for d in range(0, 11):
        R.append(E[d] / sum(E))
    return R


X = np.loadtxt('[028]wineQR(0-1).txt')
h = 0.5

X = np.delete(X, 11, axis = 1)
# 10,53,681,638,199,18
X1 = X[0:10, :]
print(X1.shape[0])
X2 = X[10:63, :]
X3 = X[63:744, :]
X4 = X[744:1382, :]
X5 = X[1382:1581, :]
X6 = X[1581:1599, :]
W = []
W.append(getRound(getE(X1)))
W.append(getRound(getE(X2)))
W.append(getRound(getE(X3)))
W.append(getRound(getE(X4)))
W.append(getRound(getE(X5)))
W.append(getRound(getE(X6)))
W=np.array(W)
W=W.flatten()
print(",".join(str(i) for i in W))
print(W)
