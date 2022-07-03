import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def getPro(theData, mean, var):
    pro = 1 / (math.sqrt(2 * math.pi) * math.sqrt(var)) * math.exp(-(theData - mean) ** 2 / (2 * var))
    return pro


X = np.loadtxt('[003]breast(0-1).txt')
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

train_index1 = [220, 23, 135, 134, 131, 156, 103, 170, 93, 185, 143, 210, 208, 128, 95, 191, 76, 126, 12, 159, 225, 224, 223, 52, 64, 214, 132, 70, 106, 56, 125, 29, 61, 25, 237, 122, 175, 145, 115, 240, 200, 151, 150, 63, 33, 138, 26, 74, 213, 110, 84, 71, 92, 162, 194, 3, 32, 14, 144, 100, 4, 232, 20, 22, 62, 109, 179, 111, 41, 171, 180, 219, 188, 206, 55, 77, 196, 140, 96, 154, 127, 226, 73, 193, 86, 31, 235, 142, 146, 107, 68, 99, 48, 97, 45, 9, 69, 83, 229, 91, 176, 13, 169, 211, 141, 139, 8, 192, 205, 152, 24, 47, 88, 168, 230, 155, 17, 231, 157, 90]
val_index1 = [30, 221, 43, 137, 87, 101, 113, 75, 49, 181, 58, 222, 5, 182, 163, 34, 218, 203, 39, 187, 164, 6, 59, 124, 195, 66, 165, 44, 27, 201, 94, 190, 104, 38, 81, 37, 82, 2, 40, 121, 72, 173, 51, 183, 50, 202, 57, 172]
test_index1 = [0, 1, 7, 10, 11, 15, 16, 18, 19, 21, 28, 35, 36, 42, 46, 53, 54, 60, 65, 67, 78, 79, 80, 85, 89, 98, 102, 105, 108, 112, 114, 116, 117, 118, 119, 120, 123, 129, 130, 133, 136, 147, 148, 149, 153, 158, 160, 161, 166, 167, 174, 177, 178, 184, 186, 189, 197, 198, 199, 204, 207, 209, 212, 215, 216, 217, 227, 228, 233, 234, 236, 238, 239]
train_index2 = [400, 234, 70, 6, 1, 156, 223, 263, 151, 41, 298, 187, 233, 437, 181, 33, 21, 45, 349, 121, 22, 110, 290, 360, 179, 104, 132, 270, 398, 68, 408, 157, 145, 429, 96, 330, 219, 339, 65, 252, 370, 302, 162, 37, 10, 385, 61, 433, 397, 248, 439, 377, 20, 171, 189, 135, 334, 353, 176, 407, 177, 306, 178, 79, 369, 438, 297, 409, 238, 214, 338, 40, 428, 170, 387, 105, 217, 130, 277, 289, 172, 345, 198, 319, 425, 29, 296, 107, 84, 266, 365, 309, 299, 427, 64, 169, 160, 325, 133, 307, 395, 288, 112, 336, 362, 421, 111, 204, 142, 230, 346, 109, 396, 423, 2, 232, 120, 25, 161, 235, 205, 186, 39, 436, 4, 114, 56, 46, 310, 446, 73, 262, 106, 453, 225, 402, 434, 159, 152, 342, 283, 216, 224, 245, 355, 90, 328, 380, 449, 447, 140, 52, 251, 54, 424, 175, 291, 280, 237, 332, 163, 190, 144, 301, 213, 352, 417, 347, 194, 226, 123, 412, 16, 249, 255, 31, 295, 278, 379, 343, 201, 221, 148, 285, 432, 85, 257, 331, 88, 63, 5, 26, 155, 38, 321, 74, 444, 203, 212, 228, 239, 344, 220, 101, 36, 300, 294, 247, 341, 356, 166, 383, 359, 415, 59, 117, 147, 71, 366, 134, 431, 18, 50, 382, 183, 378, 318, 316, 167]
val_index2 = [118, 312, 141, 354, 303, 208, 256, 261, 143, 164, 271, 77, 419, 384, 218, 125, 313, 30, 28, 75, 92, 122, 443, 392, 57, 329, 81, 138, 372, 399, 55, 150, 260, 340, 222, 324, 448, 58, 32, 279, 27, 457, 0, 286, 76, 236, 426, 11, 95, 34, 15, 420, 78, 89, 414, 17, 195, 210, 184, 124, 131, 264, 393, 149, 323, 72, 282, 174, 273, 327, 44, 430, 128, 113, 388, 127, 165, 314, 293, 153, 337, 51, 451, 116, 246, 99, 154, 442, 196, 411, 367]
test_index2 = [3, 7, 8, 9, 12, 13, 14, 19, 23, 24, 35, 42, 43, 47, 48, 49, 53, 60, 62, 66, 67, 69, 80, 82, 83, 86, 87, 91, 93, 94, 97, 98, 100, 102, 103, 108, 115, 119, 126, 129, 136, 137, 139, 146, 158, 168, 173, 180, 182, 185, 188, 191, 192, 193, 197, 199, 200, 202, 206, 207, 209, 211, 215, 227, 229, 231, 240, 241, 242, 243, 244, 250, 253, 254, 258, 259, 265, 267, 268, 269, 272, 274, 275, 276, 281, 284, 287, 292, 304, 305, 308, 311, 315, 317, 320, 322, 326, 333, 335, 348, 350, 351, 357, 358, 361, 363, 364, 368, 371, 373, 374, 375, 376, 381, 386, 389, 390, 391, 394, 401, 403, 404, 405, 406, 410, 413, 416, 418, 422, 435, 440, 441, 445, 450, 452, 454, 455, 456]


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
        W1 = W[0:3]
        for i in range(0, Class1):
            add1 = 0
            for j in range(0, 3):
                add1 += W1[j] * X[i, G1[j]]
            NewArray[i][0] = add1
        # 第1组
        W2 = W[3:9]
        for i in range(0, Class1):
            add2 = 0
            for j in range(0, 6):
                add2 += W2[j] * X[i, G2[j]]
            NewArray[i][1] = add2
        # 第2组
        W3 = W[9:10]
        for i in range(0, Class1):
            add3 = 0
            for j in range(0, 1):
                add3 += W3[j] * X[i, G3[j]]
            NewArray[i][2] = add3

        # print(NewArray)

        # 求类2的分组情况
        NewArray1 = np.ones((Class2, T + 1)) * 2
        # 第0组
        W4 = W[10:13]
        for i in range(Class1, n):
            add1 = 0
            for j in range(0, 3):
                add1 += W4[j] * X[i, G1[j]]
            NewArray1[i - Class1][0] = add1
        # 第1组
        W5 = W[13:19]
        for i in range(Class1, n):
            add2 = 0
            for j in range(0, 6):
                add2 += W5[j] * X[i, G2[j]]
            NewArray1[i - Class1][1] = add2
        # 第2组
        W6 = W[19:20]
        for i in range(Class1, n):
            add3 = 0
            for j in range(0, 1):
                add3 += W6[j] * X[i, G3[j]]
            NewArray1[i - Class1][2] = add3

        # print(NewArray1)

        # 合并两个数组，得到真正的合并数据结果
        NewArray = np.vstack((NewArray, NewArray1))


        X1 = NewArray[0:Class1, :]
        X2 = NewArray[Class1:Class1 + Class2, :]

        Data1 = X1[train_index1, :]
        Data2 = X2[train_index2, :]
        #训练集主要划分
        trainSet = np.vstack((Data1, Data2))
        Y = trainSet[:, T]
        trainSet = np.delete(trainSet, T, axis = 1)

        testSet1 = np.delete(X1[test_index1, :], T, axis = 1)
        testSet2 = np.delete(X2[test_index2, :], T, axis = 1)
        trainSet1 = np.delete(Data1, T, axis = 1)
        trainSet2 = np.delete(Data2, T, axis = 1)
        valSet1 = np.delete(X1[val_index1, :], T, axis = 1)
        valSet2 = np.delete(X2[val_index2, :], T, axis = 1)

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
