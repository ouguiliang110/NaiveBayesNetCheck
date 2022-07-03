import random
import numpy as np
import math
import matplotlib.pyplot as plt


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
train_index1 = [134, 57, 40, 102, 2, 112, 117, 73, 54, 66, 121, 92, 42, 106, 123, 97, 37, 77, 32, 14, 26, 11, 113, 122, 84, 146, 51, 94, 127, 19, 132, 99, 67, 21, 135, 64, 119, 143, 31, 0, 4, 142, 39, 86, 125, 65, 115, 61, 85, 80, 3, 103, 43, 108, 36, 90, 16, 124, 137, 8, 30, 23, 107, 52, 24, 139, 46, 72, 60, 15, 17, 70, 140]
val_index1 = [27, 35, 9, 83, 88, 95, 114, 34, 62, 29, 116, 129, 44, 128, 75, 120, 89, 71, 7, 20, 74, 81, 10, 69, 58, 145, 144, 50, 49]
test_index1 = [1, 5, 6, 12, 13, 18, 22, 25, 28, 33, 38, 41, 45, 47, 48, 53, 55, 56, 59, 63, 68, 76, 78, 79, 82, 87, 91, 93, 96, 98, 100, 101, 104, 105, 109, 110, 111, 118, 126, 130, 131, 133, 136, 138, 141]
train_index2 = [6, 27, 1, 26, 46, 18, 16, 2, 45, 5, 12, 22, 47, 43, 23, 3, 8, 4, 35, 0, 14, 36, 24, 37]
val_index2 = [29, 10, 30, 20, 42, 15, 33, 44, 11]
test_index2 = [7, 9, 13, 17, 19, 21, 25, 28, 31, 32, 34, 38, 39, 40, 41]

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

        testSet1 = np.delete(X1[test_index1, :], T, axis = 1)
        testSet2 = np.delete(X2[test_index2, :], T, axis = 1)
        trainSet1 = np.delete(Data1, T, axis = 1)
        trainSet2 = np.delete(Data2, T, axis = 1)
        valSet1 = np.delete(X1[val_index1, :], T, axis = 1)
        valSet2 = np.delete(X2[val_index2, :], T, axis = 1)

        # 求各类对应属性的均值和方差
        Mean1 = np.mean(trainSet1, axis = 0)
        Mean2 = np.mean(trainSet2, axis = 0)
        # print(Mean2)
        var1 = np.var(trainSet1, axis = 0)
        var2 = np.var(trainSet2, axis = 0)

        # 先求P(C)
        Pro1 = (train1 + 1) / (train1 + train2 + 1)
        Pro2 = (train2 + 1) / (train1 + train2 + 1)

        add = 0
        for i in range(0, val1):
            sum = 1
            for j in range(0, T):
                sum *= getPro(valSet1[i][j], Mean1[j], var1[j])
            sum1 = 1
            for j in range(0, T):
                sum1 *= getPro(valSet1[i][j], Mean2[j], var2[j])
            if Pro1 * sum >= Pro2 * sum1:
                add += 1
            elif Pro1 * sum < Pro2 * sum1:
                add += 0
        #print("第一类正确数量(总数)：",val1)
        #print(add)
        add1 = 0
        for i in range(0, val2):
            sum = 1
            for j in range(0, T):
                sum *= getPro(valSet2[i][j], Mean2[j], var2[j])
            sum1 = 1
            for j in range(0, T):
                sum1 *= getPro(valSet2[i][j], Mean1[j], var1[j])
            if Pro2 * sum >= Pro1 * sum1:
                add1 += 1
            elif Pro2 * sum < Pro1 * sum1:
                add1 += 0
        #print("第二类正确数量(总数34)：",val2)
        #print(add1)
        #print("accuracy:{:.2%}".format((add + add1) / 61))
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

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size = 14)
        plt.ylabel("fitness", size = 14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color = 'b', linewidth = 2)
        plt.show()


if __name__ == '__main__':
    NGEN = 20
    pop_size = 10
    parameters = [NGEN, pop_size]
    pso = PSO(parameters)
    pso.main()
