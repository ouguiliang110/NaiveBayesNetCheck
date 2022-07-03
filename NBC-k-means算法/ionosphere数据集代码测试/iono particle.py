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



X = np.loadtxt('[014]ionosphere(0-1).txt')
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

G1 = [6, 13, 14, 24, 30]  # 5
G2 = [5, 7, 9, 17, 21, 25, 31]# 7
G3 = [0, 1, 2, 3, 4, 8, 10, 11, 12, 15, 16, 18, 19, 20, 22, 23, 26, 27, 28, 29]# 20
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
train_index1 = [103, 35, 117, 65, 100, 5, 87, 3, 20, 58, 10, 77, 13, 122, 114, 53, 17, 52, 38, 69, 83, 120, 56, 14, 96, 25, 6, 21, 51, 105, 125, 112, 104, 107, 86, 12, 95, 68, 79, 40, 28, 46, 27, 110, 63, 97, 24, 1, 15, 9, 91, 41, 98, 54, 62, 36, 101, 123, 93, 43, 49, 99, 0]
val_index1 = [85, 78, 115, 30, 75, 106, 76, 109, 72, 32, 42, 60, 34, 33, 48, 118, 73, 45, 74, 18, 88, 90, 7, 22, 29]
test_index1 = [2, 4, 8, 11, 16, 19, 23, 26, 31, 37, 39, 44, 47, 50, 55, 57, 59, 61, 64, 66, 67, 70, 71, 80, 81, 82, 84, 89, 92, 94, 102, 108, 111, 113, 116, 119, 121, 124]
train_index2 = [142, 204, 180, 58, 190, 217, 224, 178, 78, 128, 211, 116, 206, 57, 2, 12, 174, 56, 153, 64, 97, 146, 37, 55, 100, 172, 208, 191, 89, 182, 115, 158, 85, 107, 84, 40, 162, 94, 83, 109, 139, 17, 32, 186, 18, 215, 145, 188, 147, 68, 67, 106, 168, 43, 74, 126, 133, 87, 49, 16, 8, 203, 134, 144, 160, 72, 31, 149, 23, 9, 127, 92, 200, 120, 7, 0, 189, 135, 99, 169, 95, 10, 81, 137, 101, 176, 14, 148, 53, 117, 143, 88, 63, 140, 219, 177, 121, 20, 194, 15, 207, 114, 61, 28, 50, 52, 103, 104, 222, 170, 51, 69]
val_index2 = [75, 80, 111, 155, 73, 19, 29, 45, 157, 193, 156, 118, 130, 47, 13, 102, 4, 113, 93, 202, 54, 129, 36, 212, 59, 125, 150, 163, 175, 66, 35, 21, 112, 171, 152, 218, 119, 41, 71, 3, 205, 11, 33, 216, 213]
test_index2 = [1, 5, 6, 22, 24, 25, 26, 27, 30, 34, 38, 39, 42, 44, 46, 48, 60, 62, 65, 70, 76, 77, 79, 82, 86, 90, 91, 96, 98, 105, 108, 110, 122, 123, 124, 131, 132, 136, 138, 141, 151, 154, 159, 161, 164, 165, 166, 167, 173, 179, 181, 183, 184, 185, 187, 192, 195, 196, 197, 198, 199, 201, 209, 210, 214, 220, 221, 223]

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
