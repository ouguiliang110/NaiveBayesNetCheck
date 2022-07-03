import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro


X = np.loadtxt('[021]parkinsons(0-1).txt')
m = X.shape[1]-1  # 属性数量
n = X.shape[0]  # 样本数目
K = 2  # 类标记数量
# 主要过程：分组
T = 3  # 分组数量

# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

G1=[0, 2, 8, 10, 13, 16, 17]  #7
G2=[4, 12, 14]   #3
G3=[1, 3, 5, 6, 7, 9, 11, 15, 18, 19, 20, 21]   #12


train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 确认数据集，验证集，测试集区
train_index1 = [45, 68, 140, 99, 21, 2, 5, 32, 18, 74, 39, 20, 79, 125, 94, 34, 115, 109, 136, 9, 129, 138, 4, 62, 55, 127, 77, 31, 133, 36, 76, 117, 38, 85, 54, 10, 35, 96, 57, 142, 144, 33, 130, 75, 100, 118, 47, 135, 69, 114, 46, 139, 101, 50, 58, 90, 1, 105, 11, 51, 137, 25, 93, 70, 88, 28, 104, 119, 23, 59, 89, 53, 44]
val_index1 = [13, 102, 73, 30, 7, 91, 66, 12, 82, 112, 86, 111, 97, 27, 52, 67, 123, 143, 81, 41, 126, 145, 6, 64, 98, 83, 106, 84, 95]
test_index1 = [0, 3, 8, 14, 15, 16, 17, 19, 22, 24, 26, 29, 37, 40, 42, 43, 48, 49, 56, 60, 61, 63, 65, 71, 72, 78, 80, 87, 92, 103, 107, 108, 110, 113, 116, 120, 121, 122, 124, 128, 131, 132, 134, 141, 146]
train_index2 = [40, 45, 27, 46, 29, 34, 4, 3, 2, 23, 15, 21, 11, 0, 31, 28, 32, 8, 18, 20, 42, 6, 38, 24]
val_index2 = [10, 36, 44, 39, 35, 26, 47, 7, 22]
test_index2 = [1, 5, 9, 12, 13, 14, 16, 17, 19, 25, 30, 33, 37, 41, 43]

class PSO:
    def __init__(self, parameters):
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = m * 2  # 变量个数
        self.pop_x = np.random.dirichlet(np.ones(self.var_num),size = pop_size)*10
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


        X1 = NewArray[0:Class1, :]
        X2 = NewArray[Class1:Class1 + Class2, :]

        Data1 = X1[train_index1, :]
        Data2 = X2[train_index2, :]
        #训练集主要划分
        trainSet = np.vstack((Data1, Data2))
        Y = trainSet[:, T]
        trainSet = np.delete(trainSet, T, axis = 1)

        testSet1 = np.delete(X1[test_index1, :], T, axis = 1)
        testSet2 = np.delete(X2[test_index2, :], T, axis = 1)
        trainSet1 = np.delete(Data1, T, axis = 1)
        trainSet2 = np.delete(Data2, T, axis = 1)
        valSet1 = np.delete(X1[val_index1, :], T, axis = 1)
        valSet2 = np.delete(X2[val_index2, :], T, axis = 1)
        print(valSet1)
        print(valSet2)
        clf = GaussianNB()

        clf.fit(trainSet, Y)

        C1 = clf.predict(valSet1)
        add = sum(C1 == 1)
        #print("第一类正确数量(总数):", val1)
        #print(add)
        C2 = clf.predict(valSet2)
        add1 = sum(C2 == 2)
        #print("第二类正确数量(总数)：", val2)
        #print(add1)

        #print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
        acc = (add + add1) / (val1+val2)
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
        self.ng_best=np.random.dirichlet(np.ones(m * 2), size = 1)*10
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
        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size = 14)
        plt.ylabel("fitness", size = 14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color = 'b', linewidth = 2)
        plt.show()

NGEN = 10
pop_size = 10
parameters = [NGEN, pop_size]
pso = PSO(parameters)
print("-------------")
print(pso.main())
