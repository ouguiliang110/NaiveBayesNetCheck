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
T = 7  # 分组数量

# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

G1 = [16, 23, 63, 89, 98, 124, 158] # 7
G2 = [1, 29, 31, 40, 61, 68, 70, 100, 116, 117, 118, 127, 155]# 3
G3 = [2, 3, 6, 8, 21, 24, 26, 33, 38, 51, 52, 54, 56, 57, 59, 60, 64, 71, 76, 81, 83, 84, 85, 91, 97, 99, 112, 113, 119, 120, 121, 122, 129, 131, 136, 138, 142, 143, 151, 153, 154, 160]
G4 = [13, 14, 15, 41, 45, 58, 114, 123, 135, 147]
G5 = [0, 4, 5, 9, 10, 11, 18, 19, 20, 22, 25, 27, 34, 35, 36, 37, 39, 43, 46, 47, 49, 50, 55, 65, 73, 75, 78, 80, 82, 86, 87, 88, 90, 92, 95, 96, 101, 103, 104, 107, 108, 111, 115, 130, 132, 133, 137, 139, 140, 141, 144, 148, 149, 152, 157, 159, 161, 163]
G6 = [17, 42, 48, 53, 62, 72, 74, 77, 79, 93, 94, 102, 105, 110, 125, 126, 128]
G7 = [7, 12, 28, 30, 32, 44, 66, 67, 69, 106, 109, 134, 145, 146, 150, 156, 162, 164, 165]  # 12
len1=len(G1)
len2=len(G2)
len3=len(G3)
len4=len(G4)
len5=len(G5)
len6=len(G6)
len7=len(G7)
l1=len1
l2=len1+len2
l3=len1+len2+len3
l4=len1+len2+len3+len4
l5=len1+len2+len3+len4+len5
l6=len1+len2+len3+len4+len5+len6
l7=len1+len2+len3+len4+len5+len6+len7


train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 确认数据集，验证集，测试集区
train_index1 = [108, 37, 149, 116, 29, 164, 7, 59, 161, 68, 174, 179, 151, 8, 189, 86, 16, 26, 153, 62, 176, 47, 165, 127, 60, 79, 19, 87, 201, 91, 120, 113, 28, 21, 185, 67, 117, 46, 122, 178, 194, 41, 71, 77, 180, 78, 96, 0, 12, 31, 160, 45, 76, 20, 147, 154, 49, 148, 188, 22, 1, 138, 134, 157, 88, 38, 124, 85, 183, 103, 94, 170, 56, 166, 171, 5, 4, 17, 66, 159, 123, 54, 83, 42, 119, 52, 44, 141, 146, 72, 70, 184, 35, 82, 50, 140, 92, 102, 27, 206, 155, 139, 187]
val_index1 = [158, 190, 135, 131, 169, 43, 110, 128, 39, 13, 63, 196, 129, 111, 3, 118, 89, 10, 98, 53, 58, 51, 202, 65, 55, 168, 100, 173, 18, 84, 33, 177, 115, 172, 106, 30, 167, 114, 133, 2, 61]
test_index1 = [6, 9, 11, 14, 15, 23, 24, 25, 32, 34, 36, 40, 48, 57, 64, 69, 73, 74, 75, 80, 81, 90, 93, 95, 97, 99, 101, 104, 105, 107, 109, 112, 121, 125, 126, 130, 132, 136, 137, 142, 143, 144, 145, 150, 152, 156, 162, 163, 175, 181, 182, 186, 191, 192, 193, 195, 197, 198, 199, 200, 203, 204, 205]
train_index2 = [111, 110, 247, 129, 99, 31, 171, 95, 157, 70, 214, 265, 236, 34, 144, 51, 67, 225, 133, 140, 242, 14, 43, 173, 59, 199, 112, 93, 261, 264, 156, 72, 206, 22, 50, 62, 119, 2, 132, 245, 263, 260, 121, 107, 47, 97, 82, 160, 71, 57, 116, 120, 240, 198, 28, 178, 87, 252, 106, 202, 137, 16, 266, 126, 169, 243, 33, 150, 4, 96, 45, 172, 207, 68, 48, 24, 175, 226, 190, 254, 268, 161, 149, 7, 267, 74, 256, 127, 251, 215, 177, 176, 187, 231, 248, 148, 103, 40, 73, 79, 85, 3, 42, 189, 159, 152, 77, 124, 113, 186, 9, 222, 23, 46, 15, 21, 30, 229, 210, 153, 134, 118, 61, 227, 88, 181, 44, 164, 235, 54, 12, 239, 53, 194]
val_index2 = [123, 195, 130, 211, 63, 78, 29, 18, 196, 80, 5, 204, 250, 212, 238, 101, 8, 151, 170, 223, 6, 228, 92, 36, 162, 257, 221, 83, 75, 128, 142, 255, 197, 146, 183, 185, 141, 213, 167, 26, 218, 191, 249, 49, 32, 64, 38, 37, 232, 168, 258, 94, 76]
test_index2 = [0, 1, 10, 11, 13, 17, 19, 20, 25, 27, 35, 39, 41, 52, 55, 56, 58, 60, 65, 66, 69, 81, 84, 86, 89, 90, 91, 98, 100, 102, 104, 105, 108, 109, 114, 115, 117, 122, 125, 131, 135, 136, 138, 139, 143, 145, 147, 154, 155, 158, 163, 165, 166, 174, 179, 180, 182, 184, 188, 192, 193, 200, 201, 203, 205, 208, 209, 216, 217, 219, 220, 224, 230, 233, 234, 237, 241, 244, 246, 253, 259, 262]



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

        # 第六组
        W7 = W[l6:l7]
        for i in range(0, Class1):
            add4 = 0
            for j in range(0, len7):
                add4 += W7[j] * X[i, G7[j]]
            NewArray[i][6] = add4
        # print(NewArray)
        # 求类2的分组情况
        NewArray1 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W8 = W[l7:l7 + l1]
        for i in range(Class1, n):
            add1 = 0
            for j in range(0, len1):
                add1 += W8[j] * X[i, G1[j]]
            NewArray1[i - Class1][0] = add1
        # 第1组
        W9 = W[l7 + l1:l7 + l2]
        for i in range(Class1, n):
            add2 = 0
            for j in range(0, len2):
                add2 += W9[j] * X[i, G2[j]]
            NewArray1[i - Class1][1] = add2
        # 第2组
        W10 = W[l7 + l2:l7 + l3]
        for i in range(Class1, n):
            add3 = 0
            for j in range(0, len3):
                add3 += W10[j] * X[i, G3[j]]
            NewArray1[i - Class1][2] = add3
        # 第三组
        W11 = W[l7 + l3:l7 + l4]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len4):
                add4 += W11[j] * X[i, G4[j]]
            NewArray1[i - Class1][3] = add4
        # 第三组
        W12 = W[l7 + l4:l7 + l5]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len5):
                add4 += W12[j] * X[i, G5[j]]
            NewArray1[i - Class1][4] = add4

        # 第三组
        W13 = W[l7 + l5:l7 + l6]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len6):
                add4 += W13[j] * X[i, G6[j]]
            NewArray1[i - Class1][5] = add4

        # 第三组
        W14 = W[l7 + l6:l7 + l7]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len7):
                add4 += W14[j] * X[i, G7[j]]
            NewArray1[i - Class1][6] = add4

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
