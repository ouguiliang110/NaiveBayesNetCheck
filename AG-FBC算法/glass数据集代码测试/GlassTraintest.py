import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def getPro(theData, mean, var):
    a = 1
    if (mean == 0) & (var == 0):
        return a
    else:
        pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
        return pro

X = np.loadtxt('[010]glass(0-1).txt')
# 其中有97
T = 3  # 组数量大小
m = 9  # 属性数量
n = 214  # 样本数目
K = 6  # 类标记数量
Class1 = 70
Class2 = 76
Class3 = 17
Class4 = 29
Class5 = 13
Class6 = 9
# 主要过程：分组
G1 = [6]  # 1
G2 = [1, 2, 3, 5, 7, 8]  # 6
G3 = [0, 4]  # 2


class PSO:
    def __init__(self, parameters):
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = m * 6  # 变量个数
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
        W1 = W[0:1]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, 1):
                add1 += W1[j] * X[i, G1[j]]
            NewArray1[i][0] = add1
        # 第1组
        W2 = W[1:7]
        for i in range(0, Class1):
            add2 = 0
            for j in range(0, 6):
                add2 += W2[j] * X[i, G2[j]]
            NewArray1[i][1] = add2
        # 第2组
        W3 = W[7:9]
        for i in range(0, Class1):
            add3 = 0
            for j in range(0, 2):
                add3 += W3[j] * X[i, G3[j]]
            NewArray1[i][2] = add3
        # print(NewArray1)

        # 求类2的分组情况
        NewArray2 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W4 = W[9:10]
        for i in range(Class1, Class1 + Class2):
            add1 = 0
            for j in range(0, 1):
                add1 += W4[j] * X[i, G1[j]]
            NewArray2[i - Class1][0] = add1
        # 第1组
        W5 = W[10:16]
        for i in range(Class1, Class1 + Class2):
            add2 = 0
            for j in range(0, 6):
                add2 += W5[j] * X[i, G2[j]]
            NewArray2[i - Class1][1] = add2
        # 第2组
        W6 = W[16:18]
        for i in range(Class1, Class1 + Class2):
            add3 = 0
            for j in range(0, 2):
                add3 += W6[j] * X[i, G3[j]]
            NewArray2[i - Class1][2] = add3
        # print(NewArray2)

        # 求类3的分组情况
        NewArray3 = np.ones((Class3, T + 1)) * 3
        # 第0组
        W7 = W[18:19]
        for i in range(Class1 + Class2, Class1 + Class2 + Class3):
            add1 = 0
            for j in range(0, 1):
                add1 += W7[j] * X[i, G1[j]]
            NewArray3[i - Class1 - Class2][0] = add1
        # 第1组
        W8 = W[19:25]
        for i in range(Class1 + Class2, Class1 + Class2 + Class3):
            add2 = 0
            for j in range(0, 6):
                add2 += W8[j] * X[i, G2[j]]
            NewArray3[i - Class1 - Class2][1] = add2
        # 第2组
        W9 = W[25:27]
        for i in range(Class1 + Class2, Class1 + Class2 + Class3):
            add3 = 0
            for j in range(0, 2):
                add3 += W9[j] * X[i, G3[j]]
            NewArray3[i - Class1 - Class2][2] = add3
        # print(NewArray3)

        # 求类4的分组情况
        NewArray4 = np.ones((Class4, T + 1)) * 4
        # 第0组
        W10 = W[27:28]
        for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
            add1 = 0
            for j in range(0, 1):
                add1 += W10[j] * X[i, G1[j]]
            NewArray4[i - Class1 - Class2 - Class3][0] = add1
        # 第1组
        W11 = W[28:34]
        for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
            add2 = 0
            for j in range(0, 6):
                add2 += W11[j] * X[i, G2[j]]
            NewArray4[i - Class1 - Class2 - Class3][1] = add2
        # 第2组
        W12 = W[34:36]
        for i in range(Class1 + Class2 + Class3, Class1 + Class2 + Class3 + Class4):
            add3 = 0
            for j in range(0, 2):
                add3 += W12[j] * X[i, G3[j]]
            NewArray4[i - Class1 - Class2 - Class3][2] = add3
        # print(NewArray4)
        # 求类5的分组情况
        NewArray5 = np.ones((Class5, T + 1)) * 5
        # 第0组
        W13 = W[36:37]
        for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
            add1 = 0
            for j in range(0, 1):
                add1 += W13[j] * X[i, G1[j]]
            NewArray5[i - Class1 - Class2 - Class3 - Class4][0] = add1
        # 第1组
        W14 = W[37:43]
        for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
            add2 = 0
            for j in range(0, 6):
                add2 += W14[j] * X[i, G2[j]]
            NewArray5[i - Class1 - Class2 - Class3 - Class4][1] = add2
        # 第2组
        W15 = W[43:45]
        for i in range(Class1 + Class2 + Class3 + Class4, Class1 + Class2 + Class3 + Class4 + Class5):
            add3 = 0
            for j in range(0, 2):
                add3 += W15[j] * X[i, G3[j]]
            NewArray5[i - Class1 - Class2 - Class3 - Class4][2] = add3
        # print(NewArray5)

        # 求类6的分组情况
        NewArray6 = np.ones((Class6, T + 1)) * 6
        # 第0组
        W16 = W[45:46]
        for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
            add1 = 0
            for j in range(0, 1):
                add1 += W16[j] * X[i, G1[j]]
            NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][0] = add1
        # 第1组
        W17 = W[46:52]
        for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
            add2 = 0
            for j in range(0, 6):
                add2 += W17[j] * X[i, G2[j]]
            NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][1] = add2
        # 第2组
        W18 = W[52:54]
        for i in range(Class1 + Class2 + Class3 + Class4 + Class5, Class1 + Class2 + Class3 + Class4 + Class5 + Class6):
            add3 = 0
            for j in range(0, 2):
                add3 += W18[j] * X[i, G3[j]]
            NewArray6[i - Class1 - Class2 - Class3 - Class4 - Class5][2] = add3
        # print(NewArray6)
        # 合并两个数组
        NewArray = np.vstack((NewArray1, NewArray2, NewArray3, NewArray4, NewArray5, NewArray6))
        #print(NewArray)
        Y=NewArray[:,T]
        # 去掉类标记
        NewArray = np.delete(NewArray, T, axis = 1)
        # 取训练集和测试集7：3比例，70 76 17 29 13 9  训练集149，测试65
        X1=NewArray
        # 分训练集，验证集和测试集
        # 训练集
        trainSet1 = X1[0:35, :]  # 35
        trainSet2 = X1[70:108, :]  # 38
        trainSet3 = X1[146:154, :]  # 8
        trainSet4 = X1[163:177, :]  # 14
        trainSet5 = X1[192:198, :]  # 6
        trainSet6 = X1[205:209, :]  # 4
        # trainingSet = np.vstack((Data1, Data2))
        # 验证集
        valSet1 = X1[35:49, :]  # 14
        valSet2 = X1[108:123, :]  # 15
        valSet3 = X1[154:157, :]  # 3
        valSet4 = X1[177:184, :]  # 6
        valSet5 = X1[198:201, :]  # 3
        valSet6 = X1[209:211, :]  # 2
        # 测试集
        # print(trainingSet)
        testSet1 = X1[49:70, :]  # 21
        testSet2 = X1[123:146, :]  # 23
        testSet3 = X1[157:163, :]  # 5
        testSet4 = X1[184:192, :]  # 9
        testSet5 = X1[201:205, :]  # 4
        testSet6 = X1[211:214, :]  # 3
        # testSet = np.vstack((valSet1, testSet2))
        # print(testSet)

        clf = GaussianNB()
        clf.fit(X1, Y)
        C1 = clf.predict(valSet1)
        add = sum(C1 == 1)
       # print(add)
        C2 = clf.predict(valSet2)
        add1 = sum(C2 == 2)
       # print(add1)
        C3 = clf.predict(valSet3)
        add2 = sum(C3 == 3)
       # print(add2)
        C4 = clf.predict(valSet4)
        add3 = sum(C4 == 4)
       # print(add3)
        C5 = clf.predict(valSet5)
        add4 = sum(C5 == 5)
       # print(add4)
        C6 = clf.predict(valSet6)
        add5 = sum(C6 == 6)
      #  print(add5)
      #  print("accuracy:{:.2%}".format((add + add1 + add2 + add3 + add4 + add5) / 43))

        acc = (add + add1 + add2 + add3 + add4 + add5) / 44
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
        w_orl = []
        self.ng_best = np.random.dirichlet(np.ones(m * 6), size = 1) * 10
        self.ng_best = self.ng_best.flatten()
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            orl = np.linalg.norm(self.ng_best)

            w_orl.append(orl)
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
        plt.figure(2)
        plt.title("Figure2")
        plt.xlabel("Iterators", size = 14)
        plt.ylabel("W-orl", size = 14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, w_orl, color = 'b', linewidth = 2)
        plt.show()

if __name__ == '__main__':
    NGEN = 10
    pop_size = 5
    parameters = [NGEN, pop_size]
    pso = PSO(parameters)
    pso.main()
