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
T = 4  # 分组数量

# 去掉类标记
Class1 = 0
Class2 = 0

for i in X:
    if i[m] == 1:
        Class1 = Class1 + 1
    elif i[m] == 2:
        Class2 = Class2 + 1

G1 = [7, 12, 22, 28, 32, 44, 47, 48, 53, 59, 62, 66, 67, 72, 73, 74, 77, 78, 79, 80, 105, 106, 108, 109, 110, 111, 125, 126, 134, 145, 150, 156, 159, 162, 164, 165] # 7
G2 = [1, 2, 3, 5, 6, 8, 10, 13, 14, 15, 16, 21, 23, 24, 25, 26, 27, 29, 31, 33, 34, 36, 37, 38, 39, 40, 41, 43, 45, 46, 51, 52, 54, 56, 57, 58, 60, 61, 63, 64, 68, 69, 70, 71, 76, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99, 100, 102, 103, 104, 107, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 127, 129, 130, 131, 135, 136, 137, 138, 141, 142, 143, 147, 148, 149, 151, 152, 153, 154, 155, 157, 158, 160]# 3
G3 = [4, 30, 65, 75, 144, 146, 161, 163]
G4 = [0, 9, 11, 17, 18, 19, 20, 35, 42, 49, 50, 55, 82, 93, 94, 101, 115, 128, 132, 133, 139, 140]# 12
len1=len(G1)
len2=len(G2)
len3=len(G3)
len4=len(G4)
l1=len1
l2=len1+len2
l3=len1+len2+len3
l4=len1+len2+len3+len4


train1 = int(Class1 * 0.5)
val1 = int(Class1 * 0.2)
test1 = Class1 - train1 - val1

train2 = int(Class2 * 0.5)
val2 = int(Class2 * 0.2)
test2 = Class2 - train2 - val2

# 确认数据集，验证集，测试集区
train_index1 = [80, 82, 22, 10, 17, 29, 73, 133, 119, 186, 81, 156, 149, 47, 139, 198, 175, 122, 110, 75, 50, 39, 51, 141, 78, 83, 129, 32, 151, 173, 158, 142, 20, 170, 105, 178, 103, 44, 160, 108, 164, 30, 65, 19, 182, 154, 159, 107, 102, 150, 196, 134, 41, 127, 144, 99, 124, 131, 28, 88, 188, 100, 163, 183, 9, 90, 128, 191, 146, 38, 84, 101, 0, 23, 1, 61, 66, 46, 121, 195, 71, 97, 162, 27, 69, 43, 111, 48, 104, 113, 36, 137, 53, 117, 77, 49, 35, 72, 42, 120, 26, 118, 171]
val_index1 = [112, 125, 4, 7, 172, 132, 62, 153, 64, 93, 86, 115, 181, 59, 140, 180, 189, 57, 55, 40, 94, 126, 109, 13, 58, 68, 204, 177, 169, 6, 166, 155, 12, 203, 194, 114, 143, 200, 70, 34, 14]
test_index1 = [2, 3, 5, 8, 11, 15, 16, 18, 21, 24, 25, 31, 33, 37, 45, 52, 54, 56, 60, 63, 67, 74, 76, 79, 85, 87, 89, 91, 92, 95, 96, 98, 106, 116, 123, 130, 135, 136, 138, 145, 147, 148, 152, 157, 161, 165, 167, 168, 174, 176, 179, 184, 185, 187, 190, 192, 193, 197, 199, 201, 202, 205, 206]
train_index2 = [43, 228, 247, 206, 81, 25, 232, 48, 208, 195, 189, 231, 82, 119, 218, 66, 23, 27, 229, 240, 234, 58, 204, 198, 253, 157, 174, 192, 35, 62, 83, 69, 86, 220, 87, 129, 100, 0, 169, 12, 141, 10, 227, 226, 71, 173, 239, 113, 96, 245, 133, 238, 9, 45, 233, 241, 258, 70, 91, 164, 267, 153, 56, 159, 214, 221, 184, 235, 178, 89, 42, 24, 200, 216, 163, 244, 75, 183, 72, 117, 102, 94, 168, 176, 137, 19, 148, 242, 187, 106, 139, 77, 268, 97, 202, 151, 144, 134, 171, 146, 193, 111, 55, 230, 188, 181, 152, 256, 145, 107, 260, 118, 38, 124, 7, 104, 213, 156, 251, 185, 201, 158, 78, 50, 73, 127, 179, 196, 49, 209, 15, 126, 150, 20]
val_index2 = [39, 37, 76, 217, 67, 199, 224, 54, 8, 61, 108, 98, 14, 22, 248, 109, 147, 194, 215, 142, 205, 85, 3, 122, 74, 246, 11, 44, 64, 203, 31, 237, 121, 186, 207, 30, 182, 46, 219, 172, 222, 255, 197, 131, 166, 29, 26, 254, 1, 138, 90, 149, 36]
test_index2 = [2, 4, 5, 6, 13, 16, 17, 18, 21, 28, 32, 33, 34, 40, 41, 47, 51, 52, 53, 57, 59, 60, 63, 65, 68, 79, 80, 84, 88, 92, 93, 95, 99, 101, 103, 105, 110, 112, 114, 115, 116, 120, 123, 125, 128, 130, 132, 135, 136, 140, 143, 154, 155, 160, 161, 162, 165, 167, 170, 175, 177, 180, 190, 191, 210, 211, 212, 223, 225, 236, 243, 249, 250, 252, 257, 259, 261, 262, 263, 264, 265, 266]


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
        # 第三组
        W4 = W[l3:l4]
        for i in range(0, Class1):
            add4 = 0
            for j in range(0, len4):
                add4 += W4[j] * X[i, G4[j]]
            NewArray[i][3] = add4
        # print(NewArray)

        # 求类2的分组情况
        NewArray1 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W5 = W[l4:l4+l1]
        for i in range(Class1, n):
            add1 = 0
            for j in range(0, len1):
                add1 += W5[j] * X[i, G1[j]]
            NewArray1[i - Class1][0] = add1
        # 第1组
        W6 = W[l4+l1:l4+l2]
        for i in range(Class1, n):
            add2 = 0
            for j in range(0, len2):
                add2 += W6[j] * X[i, G2[j]]
            NewArray1[i - Class1][1] = add2
        # 第2组
        W7 = W[l4+l2:l4+l3]
        for i in range(Class1, n):
            add3 = 0
            for j in range(0, len3):
                add3 += W7[j] * X[i, G3[j]]
            NewArray1[i - Class1][2] = add3
        # 第三组
        W8 = W[l4+l3:l4+l4]
        for i in range(Class1, n):
            add4 = 0
            for j in range(0, len4):
                add4 += W8[j] * X[i, G4[j]]
            NewArray1[i - Class1][3] = add4
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
