import random
import numpy as np
import math
import matplotlib.pyplot as plt


def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro


X = np.loadtxt('[017]magic(0-1).txt')
m = 10  # 属性数量
n = 1902  # 样本数目
K = 2  # 类标记数量
# 主要过程：分组
T = 3  # 分组数量
Class1 = 1219
Class2 = 683
# 随机产生多少个和为1的随机数W
G1 = [1, 2, 4, 9]
G2 = [5, 7]
G3 = [0, 3, 6, 8]


class PSO:
    def __init__(self, parameters):
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = m * 2  # 变量个数
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
        # 求类1的分组情况
        NewArray = np.ones((Class1, T + 1))
        # 第0组
        W1 = W[0:4]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, 4):
                add1 += W1[j] * X[i, G1[j]]
            NewArray[i][0] = add1
        # 第1组
        W2 = W[4:6]
        for i in range(0, Class1):
            add2 = 0
            for j in range(0, 2):
                add2 += W2[j] * X[i, G2[j]]
            NewArray[i][1] = add2
        # 第2组
        W3 = W[6:10]
        for i in range(0, Class1):
            add3 = 0
            for j in range(0, 4):
                add3 += W3[j] * X[i, G3[j]]
            NewArray[i][2] = add3

        # print(NewArray)

        # 求类2的分组情况
        NewArray1 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W4 = W[10:14]
        for i in range(Class1, n):
            add1 = 0
            for j in range(0, 4):
                add1 += W4[j] * X[i, G1[j]]
            NewArray1[i - Class1][0] = add1
        # 第1组
        W5 = W[14:16]
        for i in range(Class1, n):
            add2 = 0
            for j in range(0, 2):
                add2 += W5[j] * X[i, G2[j]]
            NewArray1[i - Class1][1] = add2
        # 第2组
        W6 = W[16:20]
        for i in range(Class1, n):
            add3 = 0
            for j in range(0, 4):
                add3 += W6[j] * X[i, G3[j]]
            NewArray1[i - Class1][2] = add3

        # print(NewArray1)

        # 合并两个数组，得到真正的合并数据结果
        NewArray = np.vstack((NewArray, NewArray1))

        # 去掉类标记
        NewArray = np.delete(NewArray, T, axis = 1)
        # 取训练集和测试集5:2:3比例 610  244  365  342  137  204
        trainSet1 = NewArray[0:610, :]
        trainSet2 = NewArray[1219:1561, :]

        # print(trainingSet)
        valSet1 = NewArray[610:854, :]
        valSet2 = NewArray[1561:1698, :]

        testSet1 = NewArray[854:1219, :]
        testSet2 = NewArray[1698:1902, :]
        # testSet = np.vstack((testSet1, testSet2))
        # print(testSet)

        # 求各类对应属性的均值和方差
        Mean1 = np.mean(trainSet1, axis = 0)
        #print(Mean1)
        Mean2 = np.mean(trainSet2, axis = 0)
        var1 = np.var(trainSet1, axis = 0)
        var2 = np.var(trainSet2, axis = 0)

        # 先求P(C)
        Pro1 = (610 + 1) / (952 + 2)
        Pro2 = (342 + 1) / (952 + 2)
        # print(Pro1)
        # print(Pro2)

        # 本次代码主要内容是这个，求P(Ai|C)

        # 统计正确数量和计算准确率
        add = 0
        for i in range(0, 244):
            sum = 1
            for j in range(0, 3):
                sum *= getPro(valSet1[i][j], Mean1[j], var1[j])
            sum1 = 1
            for j in range(0, 3):
                sum1 *= getPro(valSet1[i][j], Mean2[j], var2[j])
            if Pro1 * sum > Pro2 * sum1:
                add += 1
            elif Pro1 * sum < Pro2 * sum1:
                add += 0
        add1 = 0
        for i in range(0, 137):
            sum = 1
            for j in range(0, 3):
                sum *= getPro(valSet2[i][j], Mean2[j], var2[j])
            sum1 = 1
            for j in range(0, 3):
                sum1 *= getPro(valSet2[i][j], Mean1[j], var1[j])
            if Pro2 * sum > Pro1 * sum1:
                add1 += 1
            elif Pro2 * sum < Pro1 * sum1:
                add1 += 0
        #print("accuracy:{:.2%}".format((add + add1) / 61))
        acc = (add + add1) / 381
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
        self.ng_best=np.random.dirichlet(np.ones(m * 2), size = 1) * 100
        self.ng_best=self.ng_best.flatten()
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
        plt.xlabel("iterators", size = 14)
        plt.ylabel("fitness", size = 14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color = 'b', linewidth = 2)
        plt.show()


if __name__ == '__main__':
    NGEN = 10
    pop_size = 3
    parameters = [NGEN, pop_size]
    pso = PSO(parameters)
    pso.main()
