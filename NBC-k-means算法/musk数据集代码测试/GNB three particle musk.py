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
T = 3  # 分组数量

# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

G1 = [4, 12, 15, 17, 42, 55, 62, 69, 75, 93, 94, 128, 135, 144, 147, 162, 163]# 7
G2 = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 59, 60, 61, 63, 64, 65, 68, 70, 71, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143, 148, 149, 151, 152, 153, 154, 155, 157, 158, 160, 161]# 3
G3 = [7, 22, 28, 30, 32, 44, 58, 66, 67, 72, 79, 106, 109, 110, 134, 145, 146, 150, 156, 159, 164, 165]
len1=len(G1)
len2=len(G2)
len3=len(G3)
l1=len1
l2=len1+len2
l3=len1+len2+len3


train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 确认数据集，验证集，测试集区
train_index1 = [148, 145, 40, 10, 9, 84, 113, 155, 197, 91, 120, 138, 117, 56, 54, 22, 48, 151, 164, 150, 130, 99, 71, 23, 153, 8, 77, 64, 140, 69, 200, 11, 81, 106, 2, 166, 182, 70, 25, 80, 143, 33, 111, 5, 159, 110, 95, 46, 156, 32, 123, 188, 121, 161, 141, 51, 186, 124, 49, 3, 88, 27, 96, 7, 47, 35, 55, 185, 170, 169, 196, 128, 191, 206, 201, 195, 129, 171, 41, 168, 1, 204, 144, 26, 73, 177, 29, 89, 154, 158, 16, 38, 127, 104, 190, 181, 194, 193, 162, 19, 58, 125, 115]
val_index1 = [90, 203, 42, 21, 66, 198, 165, 98, 18, 45, 15, 87, 142, 50, 97, 78, 146, 132, 63, 137, 131, 173, 149, 157, 0, 205, 101, 172, 61, 4, 43, 176, 135, 163, 85, 83, 102, 34, 109, 119, 147]
test_index1 = [6, 12, 13, 14, 17, 20, 24, 28, 30, 31, 36, 37, 39, 44, 52, 53, 57, 59, 60, 62, 65, 67, 68, 72, 74, 75, 76, 79, 82, 86, 92, 93, 94, 100, 103, 105, 107, 108, 112, 114, 116, 118, 122, 126, 133, 134, 136, 139, 152, 160, 167, 174, 175, 178, 179, 180, 183, 184, 187, 189, 192, 199, 202]
train_index2 = [136, 223, 71, 149, 4, 264, 106, 63, 164, 171, 45, 54, 119, 35, 193, 16, 213, 165, 13, 96, 199, 101, 67, 261, 47, 156, 202, 82, 170, 166, 6, 140, 167, 112, 209, 188, 134, 79, 8, 250, 262, 128, 5, 137, 227, 155, 118, 74, 158, 93, 80, 179, 125, 7, 3, 206, 183, 66, 146, 173, 44, 92, 182, 42, 212, 122, 254, 102, 95, 23, 154, 37, 141, 32, 116, 34, 9, 160, 57, 127, 197, 58, 226, 241, 217, 150, 14, 91, 78, 30, 191, 229, 258, 28, 117, 195, 189, 153, 60, 178, 68, 65, 121, 89, 242, 73, 94, 186, 224, 69, 240, 22, 20, 214, 120, 247, 175, 100, 219, 232, 129, 131, 41, 50, 231, 87, 15, 194, 143, 208, 237, 174, 76, 144]
val_index2 = [27, 205, 103, 132, 139, 248, 177, 216, 211, 133, 185, 70, 192, 12, 138, 268, 0, 26, 252, 114, 148, 225, 236, 83, 56, 157, 207, 19, 123, 11, 52, 222, 81, 235, 255, 190, 176, 218, 49, 90, 97, 113, 38, 266, 55, 220, 204, 198, 10, 17, 99, 145, 36]
test_index2 = [1, 2, 18, 21, 24, 25, 29, 31, 33, 39, 40, 43, 46, 48, 51, 53, 59, 61, 62, 64, 72, 75, 77, 84, 85, 86, 88, 98, 104, 105, 107, 108, 109, 110, 111, 115, 124, 126, 130, 135, 142, 147, 151, 152, 159, 161, 162, 163, 168, 169, 172, 180, 181, 184, 187, 196, 200, 201, 203, 210, 215, 221, 228, 230, 233, 234, 238, 239, 243, 244, 245, 246, 249, 251, 253, 256, 257, 259, 260, 263, 265, 267]

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
        NewArray = np.ones((Class1, T + 1))
        # 第0组
        W1 = W[0:l1]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, len1):
                add1 += W1[j] * X[i, G1[j]]
            NewArray[i][0] = add1
        # 第1组
        W2 = W[l1:l2]
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

        # 求类2的分组情况
        NewArray1 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W4 = W[l3:l3 + l1]
        for i in range(Class1, n):
            add1 = 0
            for j in range(0, len1):
                add1 += W4[j] * X[i, G1[j]]
            NewArray1[i - Class1][0] = add1
        # 第1组
        W5 = W[l1 + l3:l3 + l2]
        for i in range(Class1, n):
            add2 = 0
            for j in range(0, len2):
                add2 += W5[j] * X[i, G2[j]]
            NewArray1[i - Class1][1] = add2
        # 第2组
        W6 = W[l3 + l2:l3 + l3]
        for i in range(Class1, n):
            add3 = 0
            for j in range(0, len3):
                add3 += W6[j] * X[i, G3[j]]
            NewArray1[i - Class1][2] = add3

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
    NGEN = 10
    pop_size = 100
    parameters = [NGEN, pop_size]
    pso = PSO(parameters)
    pso.main()
