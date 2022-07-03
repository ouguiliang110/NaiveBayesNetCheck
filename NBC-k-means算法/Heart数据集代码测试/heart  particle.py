import random
import numpy as np
import math
import matplotlib.pyplot as plt


def getPro(theData, mean, var):
    a=1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro=1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro



X = np.loadtxt('[012]heart(0-1).txt')
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

G1 = [1, 5, 6, 8, 9, 11, 12] # 5
G2 = [2, 3, 7, 10]# 7
G3 = [0, 4]# 20
len1=len(G1)
len2=len(G2)
len3=len(G3)
split=len1+len2+len3

train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 确认数据集，验证集，测试集区
train_index1 = [118, 36, 66, 114, 50, 30, 35, 81, 48, 20, 43, 14, 95, 4, 67, 17, 93, 15, 56, 107, 8, 101, 96, 60, 10, 0, 106, 27, 89, 42, 84, 49, 7, 119, 99, 28, 52, 73, 41, 22, 100, 12, 51, 59, 2, 37, 98, 18, 47, 92, 69, 109, 85, 113, 11, 29, 61, 5, 19, 102]
val_index1 = [6, 44, 97, 108, 62, 87, 72, 116, 105, 31, 39, 103, 1, 21, 75, 45, 55, 79, 110, 86, 70, 80, 111, 63]
test_index1 = [3, 9, 13, 16, 23, 24, 25, 26, 32, 33, 34, 38, 40, 46, 53, 54, 57, 58, 64, 65, 68, 71, 74, 76, 77, 78, 82, 83, 88, 90, 91, 94, 104, 112, 115, 117]
train_index2 = [103, 128, 67, 93, 94, 85, 141, 117, 95, 44, 91, 1, 22, 42, 60, 90, 65, 53, 116, 136, 145, 29, 119, 147, 118, 80, 71, 18, 83, 96, 30, 108, 0, 133, 45, 87, 113, 48, 62, 4, 2, 14, 98, 6, 77, 31, 15, 8, 126, 58, 56, 111, 89, 78, 132, 124, 20, 143, 130, 3, 57, 86, 142, 135, 69, 137, 73, 76, 46, 47, 55, 26, 131, 34, 12]
val_index2 = [19, 92, 121, 35, 25, 120, 101, 64, 68, 122, 33, 27, 102, 16, 5, 149, 39, 125, 13, 70, 52, 43, 50, 79, 99, 148, 123, 112, 115, 9]
test_index2 = [7, 10, 11, 17, 21, 23, 24, 28, 32, 36, 37, 38, 40, 41, 49, 51, 54, 59, 61, 63, 66, 72, 74, 75, 81, 82, 84, 88, 97, 100, 104, 105, 106, 107, 109, 110, 114, 127, 129, 134, 138, 139, 140, 144, 146]
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
        NewArray1 = np.ones((Class1, T + 1))
        # 第0组
        W1 = W[0:len1]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, len1):
                add1 += W1[j] * X[i, G1[j]]
            NewArray1[i][0] = add1
        # 第1组
        W2 = W[len1:len1 + len2]
        for i in range(0, Class1):
            add2 = 0
            for j in range(0, len2):
                add2 += W2[j] * X[i, G2[j]]
            NewArray1[i][1] = add2
        # 第2组
        W3 = W[len1 + len2:len1 + len2 + len3]
        for i in range(0, Class1):
            add3 = 0
            for j in range(0, len3):
                add3 += W3[j] * X[i, G3[j]]
            NewArray1[i][2] = add3
        # print(NewArray1)

        # 求类2的分组情况
        NewArray2 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W4 = W[split:split + len1]
        for i in range(Class1, Class1 + Class2):
            add1 = 0
            for j in range(0, len1):
                add1 += W4[j] * X[i, G1[j]]
            NewArray2[i - Class1][0] = add1
        # 第1组
        W5 = W[split + len1:split + len1 + len2]
        for i in range(Class1, Class1 + Class2):
            add2 = 0
            for j in range(0, len2):
                add2 += W5[j] * X[i, G2[j]]
            NewArray2[i - Class1][1] = add2
        # 第2组
        W6 = W[split + len1 + len2:split + len1 + len2 + len3]
        for i in range(Class1, Class1 + Class2):
            add3 = 0
            for j in range(0, len3):
                add3 += W6[j] * X[i, G3[j]]
            NewArray2[i - Class1][2] = add3
        # print(NewArray2)

        # print(NewArray1)
        # 合并两个数组，得到真正的合并数据结果

        # print(NewArray1)

        # 合并两个数组，得到真正的合并数据结果
        NewArray = np.vstack((NewArray1, NewArray2))


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
