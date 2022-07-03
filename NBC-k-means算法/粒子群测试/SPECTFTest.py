import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro


X = np.loadtxt('[024]SPECTF(0-1).txt')
# 其中有97
m = 44  # 属性数量
n = 267  # 样本数目
T = 3
K = 2  # 类标记数量
Class1 = 212
Class2 = 55
# 主要过程：分组

# 随机产生多少个和为1的随机数W
G1 = [6, 8, 10, 11, 16, 25, 31, 33, 34, 35, 38, 40, 41, 42]
G2 = [0, 3, 4, 9, 12, 14, 19, 20, 23, 24, 26, 28, 29, 32, 43]
G3 = [1, 2, 5, 7, 13, 15, 17, 18, 21, 22, 27, 30, 36, 37, 39]


class PSO:
    def __init__(self, parameters):
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = m * 2  # 变量个数
        self.pop_x = np.random.dirichlet(np.ones(self.var_num), size = pop_size) * 10
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
        # 求类1的分组情况
        NewArray = np.ones((Class1, T + 1))
        # 第0组
        W1 = W[0:14]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, 14):
                add1 += W1[j] * X[i, G1[j]]
            NewArray[i][0] = add1
        # 第1组
        W2 = W[14:29]
        for i in range(0, Class1):
            add2 = 0
            for j in range(0, 15):
                add2 += W2[j] * X[i, G2[j]]
            NewArray[i][1] = add2
        # 第2组
        W3 = W[29:44]
        for i in range(0, Class1):
            add3 = 0
            for j in range(0, 15):
                add3 += W3[j] * X[i, G3[j]]
            NewArray[i][2] = add3

        # print(NewArray)

        # 求类2的分组情况
        NewArray1 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W4 = W[44:58]
        for i in range(Class1, n):
            add1 = 0
            for j in range(0, 14):
                add1 += W4[j] * X[i, G1[j]]
            NewArray1[i - Class1][0] = add1
        # 第1组
        W5 = W[58:73]
        for i in range(Class1, n):
            add2 = 0
            for j in range(0, 15):
                add2 += W5[j] * X[i, G2[j]]
            NewArray1[i - Class1][1] = add2
        # 第2组
        W6 = W[73:88]
        for i in range(Class1, n):
            add3 = 0
            for j in range(0, 15):
                add3 += W6[j] * X[i, G3[j]]
            NewArray1[i - Class1][2] = add3
        # print(NewArray1)

        # 合并两个数组，得到真正的合并数据结果
        NewArray = np.vstack((NewArray, NewArray1))
        Y = NewArray[:, T]
        # 去掉类标记
        NewArray = np.delete(NewArray, T, axis = 1)

        # 取训练集和测试集5；2：3比例
        trainSet1 = NewArray[0:106, :]
        trainSet2 = NewArray[212:239, :]
        valSet1 = NewArray[106:148, :]
        valSet2 = NewArray[239:250, :]
        # print(trainingSet)
        testSet1 = NewArray[148:212, :]
        testSet2 = NewArray[250:267, :]
        # testSet = np.vstack((testSet1, testSet2))

        # 通过朴素贝叶斯算法得到分类器的准确率
        clf = GaussianNB()
        clf.fit(NewArray, Y)
        C1 = clf.predict(valSet1)
        add = sum(C1 == 1)
        # print(add)
        C2 = clf.predict(valSet2)
        add1 = sum(C2 == 2)
        #  print(add1)
        # print("accuracy:{:.2%}".format((add + add1) / 53))

        acc = (add + add1) / 53
        # print(acc)
        return acc

    def update_operator(self, pop_size):
        c1 = 2
        c2 = 2
        w = 0.4
        # 更新p_best和g_best
        for i in range(pop_size):
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        for i in range(pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * abs(
                self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * abs(self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护




    def main(self):
        popobj = []
        self.ng_best = np.random.dirichlet(np.ones(m * 2), size = 1) * 10
        self.ng_best = self.ng_best.flatten()
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()
            print('最好的位置：{}'.format(self.ng_best))
            print(",".join(str(i) for i in self.ng_best))
            print('最大的函数值：{}'.format(self.fitness(self.ng_best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("Iterators", size = 14)
        plt.ylabel("Accuracy", size = 14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color = 'b', linewidth = 2)
        plt.show()


if __name__ == '__main__':
    NGEN = 10
    pop_size = 3
    parameters = [NGEN, pop_size]
    pso = PSO(parameters)
    pso.main()
