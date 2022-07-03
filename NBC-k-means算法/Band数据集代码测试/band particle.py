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



X = np.loadtxt('[008]band(0-1).txt')
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

G1 = [1, 2, 3, 5] # 5
G2 = [4, 6, 8, 9, 10, 11, 13, 15, 16, 17, 20, 21, 22]# 7
G3 = [0, 7, 12, 14, 18, 19]# 20
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
train_index1 = [46, 128, 8, 225, 116, 84, 160, 167, 56, 43, 258, 218, 140, 283, 108, 54, 216, 246, 224, 288, 219, 114, 52, 220, 269, 130, 35, 30, 10, 233, 196, 237, 75, 250, 226, 86, 187, 198, 97, 217, 95, 119, 228, 49, 69, 89, 29, 123, 39, 197, 212, 182, 164, 213, 264, 282, 32, 171, 234, 280, 106, 286, 203, 168, 127, 262, 165, 48, 146, 100, 200, 207, 297, 9, 221, 107, 244, 132, 61, 151, 277, 229, 178, 239, 222, 261, 201, 159, 135, 102, 18, 138, 307, 236, 136, 94, 3, 223, 215, 99, 80, 158, 263, 122, 62, 83, 23, 26, 109, 275, 176, 205, 98, 143, 287, 79, 161, 137, 202, 129, 85, 301, 148, 289, 144, 305, 155, 310, 177, 181, 96, 186, 309, 298, 293, 68, 281, 74, 44, 34, 72, 192, 0, 183, 303, 125, 276, 53, 55, 188, 157, 5, 14, 163, 154, 40]
val_index1 = [16, 156, 147, 58, 210, 28, 257, 189, 121, 36, 37, 139, 231, 42, 295, 65, 260, 184, 254, 242, 174, 235, 90, 195, 256, 149, 88, 47, 199, 179, 248, 170, 290, 240, 243, 273, 292, 193, 77, 173, 22, 103, 166, 291, 93, 185, 4, 19, 64, 66, 33, 27, 227, 134, 112, 204, 268, 60, 124, 172, 169, 59]
test_index1 = [1, 2, 6, 7, 11, 12, 13, 15, 17, 20, 21, 24, 25, 31, 38, 41, 45, 50, 51, 57, 63, 67, 70, 71, 73, 76, 78, 81, 82, 87, 91, 92, 101, 104, 105, 110, 111, 113, 115, 117, 118, 120, 126, 131, 133, 141, 142, 145, 150, 152, 153, 162, 175, 180, 190, 191, 194, 206, 208, 209, 211, 214, 230, 232, 238, 241, 245, 247, 249, 251, 252, 253, 255, 259, 265, 266, 267, 270, 271, 272, 274, 278, 279, 284, 285, 294, 296, 299, 300, 302, 304, 306, 308, 311]
train_index2 = [225, 78, 186, 174, 202, 35, 210, 90, 119, 217, 211, 82, 89, 11, 147, 203, 171, 187, 73, 100, 151, 87, 125, 114, 177, 91, 103, 149, 141, 156, 148, 108, 77, 215, 131, 52, 201, 112, 4, 93, 224, 44, 107, 33, 47, 101, 176, 183, 150, 53, 214, 142, 140, 40, 25, 164, 207, 160, 14, 127, 83, 63, 1, 109, 30, 5, 7, 204, 17, 16, 168, 65, 122, 123, 191, 64, 139, 19, 189, 49, 161, 178, 169, 163, 155, 98, 74, 195, 8, 2, 182, 137, 136, 39, 206, 75, 200, 72, 194, 48, 54, 116, 153, 185, 205, 188, 130, 128, 219, 76, 199, 66, 196, 12]
val_index2 = [70, 126, 227, 170, 165, 118, 55, 172, 22, 102, 92, 97, 67, 41, 132, 50, 117, 146, 69, 51, 59, 21, 193, 198, 181, 134, 197, 226, 124, 166, 58, 162, 111, 88, 32, 94, 9, 45, 212, 84, 26, 208, 15, 138, 29]
test_index2 = [0, 3, 6, 10, 13, 18, 20, 23, 24, 27, 28, 31, 34, 36, 37, 38, 42, 43, 46, 56, 57, 60, 61, 62, 68, 71, 79, 80, 81, 85, 86, 95, 96, 99, 104, 105, 106, 110, 113, 115, 120, 121, 129, 133, 135, 143, 144, 145, 152, 154, 157, 158, 159, 167, 173, 175, 179, 180, 184, 190, 192, 209, 213, 216, 218, 220, 221, 222, 223]

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
