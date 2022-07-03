import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

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

def getRandom(num):
    Ran = np.random.dirichlet(np.ones(num), size = 1)
    Ran = Ran.flatten()
    return Ran


X = np.loadtxt('[018]musk01(0-1).txt')
# 主要的参数设置
n1 = 0.0001
n2 = 0.0001
n3 = 0.00001
N = 100
b = 1  # β值大小
T = 20  # 组数量大小
m = X.shape[1]-1  # 属性数量
n = X.shape[0]  # 样本数目 其中第一类212，第二类55
print(X.shape)

SetClass=set(X[:,m])
SetClass=list(map(int,SetClass))
print(SetClass)
K = len(SetClass)  # 类标记数量

Z = np.zeros((K, m))
U = np.zeros((n, K))

#print(X.shape)
# print(U)
# print(X)

# 创建空数组




#处理数据集20是根据最大类数来处理得到的
newarray=[np.zeros(shape=[0,m+1])]*20
for i in X:
    for j in SetClass:
        if i[m] == j:
            newarray[j] = np.vstack((newarray[j], i))

NewArray=np.zeros(shape=[0,m+1])
for i in SetClass:
    NewArray=np.vstack((NewArray,newarray[i]))
print(NewArray)
print(NewArray.shape)


#print(newarray)

#统计各类数量
NumClass=[0]*K
# 初始化U
p=0
for i in X:
    for j in range(0,K):
        if i[m]==SetClass[j]:
            U[p][j-1]=1
            NumClass[j]=NumClass[j]+1
    p=p+1
#print(NumClass)
'''
train=[]
val=[]
test=[]
for i in range(0,K):
    train.append(int(NumClass[i] * 0.5))
    val.append(int(NumClass[i] * 0.2))
    test.append(NumClass[i] - train[i] - val[i])
print(train)
print(val)
print(test)
train_index=[]
val_index=[]
test_index=[]
for i in range(0,K):
    idx=np.random.choice(np.arange(NumClass[i]), size = train[i], replace = False)
    train_index.append(np.array(idx))
    val_index.append(np.random.choice(np.delete(np.arange(NumClass[i]), train_index[i]), size =val[i], replace = False))
    test_index.append(np.delete(np.arange(NumClass[i]), np.append(train_index[i], val_index[i])))
print(train_index)
print(val_index)
print(test_index)

'''


#print(p1)
#print(p)
X = np.delete(NewArray, m, axis = 1)

# 取得除去类列外的剩余值，这里主要求Z
# print(X)
# 初始化Z矩阵
# 初始化Z矩阵

#注意类和下标之间的转换关系
p=0
for i in SetClass:
    temp=np.delete(newarray[i],m,axis = 1)
    Z[p]=np.mean(temp,axis = 0)
    #print(Z[p])
    p = p + 1

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
#
Group=[]
for t in range(0,T):
    print("第" + repr(t) + "组")
    list1 = []
    for j in range(0,m):
           if G[j,t]==1:
               list1.append(j)
           else:
               continue
    Group.append(list1)
    print(list1)
    print('\n')
print("---------------------")
for i in range(0,len(Group)):
    print(Group[i])

#根据粒子群优化得出最优解
train=[]
trainNum=0
val=[]
valNum=0
test=[]
testNum=0
for i in range(0,K):
    train.append(int(NumClass[i] * 0.5))
    trainNum+=int(NumClass[i] * 0.5)

    val.append(int(NumClass[i] * 0.2))
    valNum+=int(NumClass[i] * 0.2)

    test.append(NumClass[i] - train[i] - val[i])
    testNum+=NumClass[i] - train[i] - val[i]

train_index=[]
val_index=[]
test_index=[]
for i in range(0,K):
    idx=np.random.choice(np.arange(NumClass[i]), size = train[i], replace = False)
    train_index.append(np.array(idx))
    val_index.append(np.random.choice(np.delete(np.arange(NumClass[i]), train_index[i]), size =val[i], replace = False))
    test_index.append(np.delete(np.arange(NumClass[i]), np.append(train_index[i], val_index[i])))


class PSO:
    def __init__(self, parameters):
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = m # 变量个数
        self.pop_x = np.random.dirichlet(np.ones(self.var_num),size = pop_size)*100
        self.pop_v = np.zeros((self.pop_size, self.var_num))
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置

        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # 存储最优的个体
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, W):
        TheNewArray = []

        for i in range(0, K):
            getArray = np.ones((NumClass[i], T + 1)) * SetClass[i]
            p1=0
            for j in range(0, T):
                W1 = W[p1:p1+len(Group[j])]
                for a in range(0, NumClass[i]):
                    add = 0
                    for b in range(0, len(Group[j])):
                        add += W1[b] * NewArray[a, Group[j][b]]
                    getArray[a][j] = add
                p1 = p1 + len(Group[j])
            TheNewArray.append(getArray)
        NewArrayQ = np.zeros((0, T + 1))
        for i in range(0, len(TheNewArray)):
            NewArrayQ = np.vstack((NewArrayQ, TheNewArray[i]))
        #print(NewArrayQ)

        dividX = []
        p2 = 0
        for i in range(0, K):
            dividX.append(NewArrayQ[p2:p2 + NumClass[i], :])
            p2 = p2 + NumClass[i]

        trainSet = []
        for i in range(0, K):
            trainSet.append(dividX[i][train_index[i], :])
        TrainSet = np.zeros((0, T + 1))
        for i in range(0, K):
            TrainSet = np.vstack((TrainSet, trainSet[i]))
        Y = TrainSet[:, T]
        TrainSet=np.delete(TrainSet,T,axis = 1)
        for i in range(0, K):
            trainSet[i] = np.delete(trainSet[i], T, axis = 1)

        testSet = []
        for i in range(0, K):
            testSet.append(np.delete(dividX[i][test_index[i], :], T, axis = 1))
        valSet = []
        for i in range(0, K):
            valSet.append(np.delete(dividX[i][val_index[i], :], T, axis = 1))
        clf = GaussianNB()

        clf.fit(TrainSet, Y)

        correct=0
        for i in range(0,K):
            C=clf.predict(valSet[i])
            correct+=sum(C==SetClass[i])

        #print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
        acc = correct / valNum
        return acc

    def update_operator(self, pop_size):
        c1 = 2
        c2 = 2
        w = 0.4
        for i in range(pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * abs(
                    self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * abs(self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护

            # 更新p_best和g_best
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]

    def main(self):
        popobj = []
        self.ng_best=np.random.dirichlet(np.ones(m), size = 1)*100
        self.ng_best=self.ng_best.flatten()
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()
            print('最好的位置：{}'.format(self.ng_best))
            print(list(self.ng_best))
            print('最大的函数值：{}'.format(self.fitness(self.ng_best)))

        print("---- End of (successful) Searching ----")
        return self.ng_best



TheNewArray = []
p1 = 0
for i in range(0, K):
    getArray = np.ones((NumClass[i], T + 1)) * SetClass[i]
    p1 = 0
    for j in range(0, T):
        W1 = W[p1:p1 + len(Group[j])]
        for a in range(0, NumClass[i]):
            add = 0
            for b in range(0, len(Group[j])):
                add += W1[b] * NewArray[a, Group[j][b]]
            getArray[a][j] = add
        p1 = p1 + len(Group[j])
    TheNewArray.append(getArray)
NewArrayQ = np.zeros((0, T + 1))
for i in range(0, len(TheNewArray)):
    NewArrayQ = np.vstack((NewArrayQ, TheNewArray[i]))
print(NewArrayQ)

dividX = []
p2 = 0
for i in range(0, K):
    dividX.append(NewArrayQ[p2:p2 + NumClass[i], :])
    p2 = p2 + NumClass[i]

trainSet = []
for i in range(0, K):
    trainSet.append(dividX[i][train_index[i], :])
TrainSet = np.zeros((0, T + 1))
for i in range(0, K):
    TrainSet = np.vstack((TrainSet, trainSet[i]))
Y = TrainSet[:, T]
TrainSet = np.delete(TrainSet, T, axis = 1)
for i in range(0, K):
    trainSet[i] = np.delete(trainSet[i], T, axis = 1)

testSet = []
for i in range(0, K):
    testSet.append(np.delete(dividX[i][test_index[i], :], T, axis = 1))
valSet = []
for i in range(0, K):
    valSet.append(np.delete(dividX[i][val_index[i], :], T, axis = 1))
clf = GaussianNB()

clf.fit(TrainSet, Y)

correct=0
for i in range(0,K):
    C=clf.predict(testSet[i])
    print(C)
    print(SetClass[i])
    correct+=sum(C==SetClass[i])
    print(sum(C==SetClass[i]))
    print(correct)
    #print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
testacc = correct / testNum
print("test accuracy:{:.2%}".format(testacc))
print("---------------------------")
correct1=0
for i in range(0,K):
    C=clf.predict(trainSet[i])
    print(C)
    print(SetClass[i])
    correct1+=sum(C==SetClass[i])
    print(sum(C == SetClass[i]))
    print(correct1)
    #print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
trainacc = correct1 / trainNum
print("train accuracy:{:.2%}".format(trainacc))