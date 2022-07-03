import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro


X = np.loadtxt('[018]musk01(0-1).txt')
m = X.shape[1]-1  # 属性数量
n = X.shape[0]  # 样本数目
K = 2  # 类标记数量
# 主要过程：分组
T = 6  # 分组数量

# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

G1 = [16, 23, 29, 31, 40, 61, 63, 89, 98, 100, 116, 117, 124, 155, 158] # 7
G2 = [1, 6, 21, 24, 33, 38, 52, 56, 64, 68, 70, 76, 83, 84, 85, 99, 113, 118, 119, 120, 122, 127, 131, 143, 153]# 3
G3 = [13, 15, 41, 45, 53, 58, 69, 102, 135, 147]
G4 = [2, 3, 5, 8, 22, 25, 26, 44, 51, 54, 57, 59, 60, 71, 81, 91, 92, 95, 96, 97, 112, 114, 121, 123, 129, 136, 138, 142, 149, 151, 154, 160]
G5 = [0, 4, 9, 10, 11, 14, 17, 18, 19, 20, 27, 34, 35, 36, 37, 39, 42, 43, 46, 47, 48, 49, 50, 55, 62, 65, 72, 73, 74, 75, 77, 78, 80, 82, 86, 87, 88, 90, 93, 94, 101, 103, 104, 105, 107, 108, 110, 111, 115, 125, 126, 128, 130, 132, 133, 137, 139, 140, 141, 144, 148, 152, 157, 159, 161, 163]
G6 = [7, 12, 28, 30, 32, 66, 67, 79, 106, 109, 134, 145, 146, 150, 156, 162, 164, 165]  # 12
len1=len(G1)
len2=len(G2)
len3=len(G3)
len4=len(G4)
len5=len(G5)
len6=len(G6)
l1=len1
l2=len1+len2
l3=len1+len2+len3
l4=len1+len2+len3+len4
l5=len1+len2+len3+len4+len5
l6=len1+len2+len3+len4+len5+len6


train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 确认数据集，验证集，测试集区
train_index1 = [160, 19, 112, 78, 66, 46, 180, 158, 16, 128, 164, 184, 99, 23, 157, 116, 83, 115, 175, 151, 3, 206, 79, 120, 72, 132, 144, 10, 162, 52, 111, 82, 45, 48, 30, 69, 60, 21, 80, 44, 204, 43, 76, 192, 167, 11, 113, 186, 6, 200, 94, 177, 0, 182, 203, 75, 171, 89, 196, 173, 194, 14, 198, 107, 202, 124, 24, 61, 197, 188, 91, 42, 77, 106, 7, 85, 170, 147, 159, 152, 174, 185, 31, 13, 68, 96, 114, 121, 138, 25, 166, 178, 5, 102, 59, 201, 26, 101, 119, 49, 187, 17, 191]
val_index1 = [41, 108, 143, 9, 136, 39, 81, 154, 37, 149, 51, 176, 50, 134, 155, 1, 35, 137, 98, 193, 36, 34, 126, 63, 127, 105, 123, 169, 54, 179, 142, 133, 62, 131, 2, 33, 28, 97, 181, 110, 29]
test_index1 = [4, 8, 12, 15, 18, 20, 22, 27, 32, 38, 40, 47, 53, 55, 56, 57, 58, 64, 65, 67, 70, 71, 73, 74, 84, 86, 87, 88, 90, 92, 93, 95, 100, 103, 104, 109, 117, 118, 122, 125, 129, 130, 135, 139, 140, 141, 145, 146, 148, 150, 153, 156, 161, 163, 165, 168, 172, 183, 189, 190, 195, 199, 205]
train_index2 = [138, 88, 99, 30, 108, 256, 77, 155, 257, 135, 93, 87, 245, 162, 171, 27, 260, 96, 146, 126, 150, 251, 105, 15, 229, 127, 81, 225, 258, 65, 18, 137, 190, 51, 9, 205, 113, 210, 166, 202, 35, 49, 22, 139, 72, 174, 145, 228, 45, 98, 97, 243, 160, 47, 112, 70, 236, 117, 84, 56, 76, 25, 67, 259, 250, 263, 101, 114, 54, 185, 249, 59, 209, 168, 233, 159, 31, 238, 44, 39, 23, 226, 62, 170, 8, 133, 2, 89, 82, 52, 177, 79, 80, 230, 254, 106, 46, 219, 182, 136, 240, 119, 156, 21, 196, 115, 107, 220, 262, 267, 28, 178, 242, 86, 75, 134, 264, 33, 143, 32, 206, 3, 237, 172, 13, 74, 69, 130, 235, 12, 124, 100, 152, 48]
val_index2 = [157, 41, 26, 261, 116, 193, 125, 6, 43, 218, 36, 34, 118, 188, 5, 203, 181, 241, 214, 50, 149, 231, 147, 239, 163, 248, 200, 92, 17, 223, 194, 73, 253, 38, 175, 90, 186, 207, 40, 140, 199, 268, 201, 142, 24, 227, 63, 60, 103, 53, 102, 222, 95]
test_index2 = [0, 1, 4, 7, 10, 11, 14, 16, 19, 20, 29, 37, 42, 55, 57, 58, 61, 64, 66, 68, 71, 78, 83, 85, 91, 94, 104, 109, 110, 111, 120, 121, 122, 123, 128, 129, 131, 132, 141, 144, 148, 151, 153, 154, 158, 161, 164, 165, 167, 169, 173, 176, 179, 180, 183, 184, 187, 189, 191, 192, 195, 197, 198, 204, 208, 211, 212, 213, 215, 216, 217, 221, 224, 232, 234, 244, 246, 247, 252, 255, 265, 266]


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
        # 求类1的分组情况
        # 求类1的分组情况
        # 求类1的分组情况
        NewArray = np.ones((Class1, T + 1))
        # 第0组
        W1 = W[0:l1]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, len1):
                add1 += W1[j] * X[i, G1[j]]
            NewArray[i][0] = add1
        # 第1组
        W2 = W[l1:l1 + l2]
        for i in range(0, Class1):
            add2 = 0
            for j in range(0, len2):
                add2 += W2[j] * X[i, G2[j]]
            NewArray[i][1] = add2
        # 第2组
        W3 = W[l2:l3]
        for i in range(0, Class1):
            add3 = 0
            for j in range(0, len3):
                add3 += W3[j] * X[i, G3[j]]
            NewArray[i][2] = add3
        # 第三组
        W4 = W[l3:l4]
        for i in range(0, Class1):
            add4 = 0
            for j in range(0, len4):
                add4 += W4[j] * X[i, G4[j]]
            NewArray[i][3] = add4
        # 第四组
        W5 = W[l4:l5]
        for i in range(0, Class1):
            add4 = 0
            for j in range(0, len5):
                add4 += W5[j] * X[i, G5[j]]
            NewArray[i][4] = add4
        # 第五组
        W6 = W[l5:l6]
        for i in range(0, Class1):
            add4 = 0
            for j in range(0, len6):
                add4 += W6[j] * X[i, G6[j]]
            NewArray[i][5] = add4

        # 求类2的分组情况
        NewArray1 = np.ones((Class2, T + 1)) * 2
        W8 = W[l6:l6 + l1]
        for i in range(Class1, n):
            add1 = 0
            for j in range(0, len1):
                add1 += W8[j] * X[i, G1[j]]
            NewArray1[i - Class1][0] = add1
        # 第1组
        W9 = W[l6 + l1:l6 + l2]
        for i in range(Class1, n):
            add2 = 0
            for j in range(0, len2):
                add2 += W9[j] * X[i, G2[j]]
            NewArray1[i - Class1][1] = add2
        # 第2组
        W10 = W[l6 + l2:l6 + l3]
        for i in range(Class1, n):
            add3 = 0
            for j in range(0, len3):
                add3 += W10[j] * X[i, G3[j]]
            NewArray1[i - Class1][2] = add3
        # 第三组
        W11 = W[l6 + l3:l6 + l4]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len4):
                add4 += W11[j] * X[i, G4[j]]
            NewArray1[i - Class1][3] = add4
        # 第三组
        W12 = W[l6 + l4:l6 + l5]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len5):
                add4 += W12[j] * X[i, G5[j]]
            NewArray1[i - Class1][4] = add4

        # 第三组
        W13 = W[l6 + l5:l6 + l6]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len6):
                add4 += W13[j] * X[i, G6[j]]
            NewArray1[i - Class1][5] = add4

        NewArray = np.vstack((NewArray, NewArray1))
        # 随机抽取样本训练集和测试集样本

        X1 = NewArray[0:Class1, :]
        X2 = NewArray[Class1:Class1 + Class2, :]

        Data1 = X1[train_index1, :]
        Data2 = X2[train_index2, :]

        trainSet = np.vstack((Data1, Data2))
        Y = trainSet[:, T]
        trainSet = np.delete(trainSet, T, axis = 1)

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
