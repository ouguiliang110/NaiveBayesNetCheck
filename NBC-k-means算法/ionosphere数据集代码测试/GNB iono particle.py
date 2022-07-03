import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
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
train_index1 = [113, 10, 26, 3, 87, 101, 34, 117, 105, 79, 22, 112, 104, 76, 115, 92, 55, 111, 60, 38, 29, 59, 53, 54, 123, 57, 93, 28, 52, 97, 80, 78, 90, 35, 16, 68, 13, 48, 82, 72, 15, 47, 9, 51, 30, 1, 75, 32, 17, 70, 18, 88, 84, 96, 41, 74, 5, 24, 8, 58, 33, 100, 64]
val_index1 = [0, 116, 62, 14, 36, 65, 27, 66, 99, 67, 69, 125, 91, 12, 56, 83, 20, 46, 25, 86, 73, 109, 2, 40, 6]
test_index1 = [4, 7, 11, 19, 21, 23, 31, 37, 39, 42, 43, 44, 45, 49, 50, 61, 63, 71, 77, 81, 85, 89, 94, 95, 98, 102, 103, 106, 107, 108, 110, 114, 118, 119, 120, 121, 122, 124]
train_index2 = [187, 191, 172, 36, 107, 167, 175, 143, 129, 70, 202, 25, 161, 48, 222, 141, 160, 67, 18, 162, 140, 134, 40, 156, 193, 32, 188, 154, 168, 163, 111, 19, 93, 63, 82, 87, 182, 95, 203, 194, 147, 206, 9, 137, 61, 122, 217, 165, 209, 46, 89, 120, 76, 177, 101, 51, 49, 117, 151, 148, 21, 68, 138, 58, 224, 127, 26, 62, 71, 213, 110, 27, 199, 30, 190, 17, 92, 152, 11, 142, 205, 3, 136, 185, 86, 220, 208, 59, 196, 197, 52, 35, 91, 39, 119, 2, 5, 121, 181, 123, 170, 6, 179, 53, 16, 192, 212, 173, 133, 184, 88, 72]
val_index2 = [180, 69, 85, 135, 98, 57, 223, 195, 178, 1, 116, 139, 104, 108, 153, 159, 64, 100, 94, 42, 83, 169, 47, 65, 211, 60, 216, 219, 33, 126, 34, 99, 24, 20, 79, 150, 115, 54, 75, 14, 31, 13, 74, 12, 102]
test_index2 = [0, 4, 7, 8, 10, 15, 22, 23, 28, 29, 37, 38, 41, 43, 44, 45, 50, 55, 56, 66, 73, 77, 78, 80, 81, 84, 90, 96, 97, 103, 105, 106, 109, 112, 113, 114, 118, 124, 125, 128, 130, 131, 132, 144, 145, 146, 149, 155, 157, 158, 164, 166, 171, 174, 176, 183, 186, 189, 198, 200, 201, 204, 207, 210, 214, 215, 218, 221]

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
