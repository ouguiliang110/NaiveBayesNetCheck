import math
import random
from minepy import MINE
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA

random.seed(0)


##首先实现几个工具函数:
def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill = 0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


##定义sigmod函数和它的导数:
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


'''
定义BPNeuralNetwork类， 使用三个列表维护输入层，
隐含层和输出层神经元， 
列表中的元素代表对应神经元当前的输出值.使用两个二维列表以邻接矩阵的形式维护输入层与隐含层， 
隐含层与输出层之间的连接权值， 通过同样的形式保存矫正矩阵
'''


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    # 设置隐藏层 输入层，隐藏层，输出层
    def setup(self, ni, nh, no):
        self.input_n = ni + 1  # 输入层
        self.hidden_n = nh  # 隐藏层
        self.output_n = no  # 输出层
        self.hidden_matrix = []
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        print(self.input_weights)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        print(self.output_weights)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    # 定义predict方法进行一次前馈， 并返回输出:
    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]

        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)

        self.hidden_matrix.append(self.hidden_cells)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:], self.hidden_cells[:]

    # 定义back_propagate方法定义一次反向传播和更新权值的过程， 并返回最终预测误差
    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error  获得输出层的误差
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error  获得隐藏层的误差
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights   更新输出层的权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights   更新输入层的权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error       获得整个误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    # 返回属性相关性
    def getHiddenRelation(self, HiddenLayer):
        HiddenLayer = np.array(HiddenLayer)
        mine = MINE(alpha = 0.6, c = 15)
        relation = 0
        for i in range(self.hidden_n):
            for j in range(i + 1, self.hidden_n):
                # mine.compute_score(HiddenLayer[:, i], HiddenLayer[:, j])
                # relation=relation + mine.mic()
                a = pearsonr(HiddenLayer[:, i], HiddenLayer[:, j])
                relation = relation + abs(a[0])
        return relation

    # 定义train方法控制迭代， 该方法可以修改最大迭代次数， 学习率λ， 矫正率μ三个参数.
    def getTheBestAccurate(self, HiddenMatrix):
        New_data = HiddenMatrix
        clf = GaussianNB()
        clf.fit(New_data[:, :-1], New_data[:, -1])

        correct = 0
        C = clf.predict(New_data[:, :-1])
        for i in range(0, HiddenMatrix.shape[0]):
            if C[i] == New_data[:, -1][i]:
                correct = correct + 1
        return correct / HiddenMatrix.shape[0]

    def train(self, cases, labels, theLabel, limit = 500, learn = 0.05, correct = 0.1):
        E = 10000
        acc = 0
        Min_matrix = []
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            Hidden_matrix = []
            for case in cases:
                # 返回隐藏矩阵
                get = self.predict(case)
                Hidden_matrix.append(get[1])
            # 新数组的相关性分析
            hidden_correlation = self.getHiddenRelation(Hidden_matrix)

            New_data = []
            for i in range(0, np.array(Hidden_matrix).shape[0]):
                New_data.append(np.append(Hidden_matrix[i], theLabel[i]))
            New_data = np.array(New_data)
            if E > (hidden_correlation + error):
                E = hidden_correlation + error
                acc1 = self.getTheBestAccurate(New_data)
                if acc1 >= acc:
                    Min_matrix = Hidden_matrix
                    acc = acc1
                print(acc1)
            print(E)
        return Min_matrix

    # return error
    def getAccurate(self, HiddenMatrix):
        HiddenMatrix=np.array(HiddenMatrix)

        m = HiddenMatrix.shape[1] - 1
        n = HiddenMatrix.shape[0]

        vector_data = HiddenMatrix[:, :-1]
        # 提取label类别
        label_data = HiddenMatrix[:, -1]

        SetClass = set(HiddenMatrix[:, m])
        SetClass = list(map(int, SetClass))
        # print(SetClass)
        K = len(SetClass)  # 类标记数量

        # 合并同类样本，顺序排列
        newarray = [np.zeros(shape = [0, m + 1])] * 20
        for i in HiddenMatrix:
            for j in SetClass:
                if i[m] == j:
                    newarray[j] = np.vstack((newarray[j], i))

        NewArray = np.zeros(shape = [0, m + 1])
        for i in SetClass:
            NewArray = np.vstack((NewArray, newarray[i]))
        HiddenMatrix = NewArray
        '''
        #对类进行排序，即同类归类到一个中，方便比较
        array1 = np.zeros(shape = (0, n))
        for n in range(0, m):
            k = 25
            d1 = pd.cut(vector_data[:, n], k, labels = range(k))
            array1 = np.vstack((array1, d1))
        array1 = np.vstack((array1, label_data))
        X1 = array1.T
        print(X1)
        array = X1
        '''

        NumClass = [0] * K
        # 初始化U
        p = 0
        for i in HiddenMatrix:
            for j in range(0, K):
                if i[m] == SetClass[j]:
                    NumClass[j] = NumClass[j] + 1
            p = p + 1
        # print(NumClass)

        arr_train = []
        arr_test = []

        for k in range(0, 10):
            train = []
            trainNum = 0
            test = []
            testNum = 0
            for i in range(0, K):
                train.append(int(NumClass[i] * 0.7))
                trainNum += int(NumClass[i] * 0.7)

                test.append(NumClass[i] - train[i])
                testNum += NumClass[i] - train[i]

            train_index = []
            test_index = []
            for i in range(0, K):
                idx = np.random.choice(np.arange(NumClass[i]), size = train[i], replace = False)
                train_index.append(np.array(idx))
                # print(train_index)
                # val_index.append(np.random.choice(np.delete(np.arange(NumClass[i]), train_index[i]), size = val[i], replace = False))
                test_index.append(np.delete(np.arange(NumClass[i]), train_index[i]))
                # print(test_index)
            dividX = []
            p2 = 0
            for i in range(0, K):
                dividX.append(HiddenMatrix[p2:p2 + NumClass[i], :])
                p2 = p2 + NumClass[i]

            trainSet = []
            for i in range(0, K):
                trainSet.append(dividX[i][train_index[i], :])
            TrainSet = np.zeros((0, m + 1))
            for i in range(0, K):
                TrainSet = np.vstack((TrainSet, trainSet[i]))
            # print(TrainSet)
            Y = TrainSet[:, m]
            # print(Y)
            TrainSet = np.delete(TrainSet, m, axis = 1)
            for i in range(0, K):
                trainSet[i] = np.delete(trainSet[i], m, axis = 1)

            testSet = []
            for i in range(0, K):
                testSet.append(np.delete(dividX[i][test_index[i], :], m, axis = 1))
            # print(testSet)
            # print(testSet)

            # print(valSet)
            clf1 = GaussianNB()
            clf1.fit(TrainSet, Y)

            correct = 0
            for i in range(0, K):
                C = clf1.predict(testSet[i])
                # print(C)
                # print(SetClass[i])
                correct += sum(C == SetClass[i])
                # print(sum(C == SetClass[i]))
                # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
            testacc = correct / testNum
            arr_test.append(testacc)
            # print("test accuracy:{:.2%}".format(testacc))
            # print("---------------------------")
            correct1 = 0
            for i in range(0, K):
                C = clf1.predict(trainSet[i])
                # print(C)
                # print(SetClass[i])
                correct1 += sum(C == SetClass[i])
                # print(sum(C == SetClass[i]))
                # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
            arr_train.append(correct1 / trainNum)
            # print("train accuracy:{:.2%}".format(trainacc))

        # 求标准差
        ##print(arr_train)

        arr_mean = np.mean(arr_train)
        arr_std = np.std(arr_train, ddof = 1)

        arr_mean1 = np.mean(arr_test)
        arr_std1 = np.std(arr_test, ddof = 1)

        # print("训练集平均标准差", arr_mean, arr_std)
        # print("测试集平均标准差", arr_mean1, arr_std1)

        return arr_mean, arr_std, arr_mean1, arr_std1

    def getTheAccurate(self,HiddenMatrix):

        m = HiddenMatrix.shape[1] - 1
        n = HiddenMatrix.shape[0]

        vector_data = HiddenMatrix[:, :-1]
        # 提取label类别
        label_data = HiddenMatrix[:, -1]

        SetClass = set(HiddenMatrix[:, m])
        SetClass = list(map(int, SetClass))
        # print(SetClass)
        K = len(SetClass)  # 类标记数量

        # 合并同类样本，顺序排列
        newarray = [np.zeros(shape = [0, m + 1])] * 20
        for i in HiddenMatrix:
            for j in SetClass:
                if i[m] == j:
                    newarray[j] = np.vstack((newarray[j], i))

        NewArray = np.zeros(shape = [0, m + 1])
        for i in SetClass:
            NewArray = np.vstack((NewArray, newarray[i]))
        HiddenMatrix = NewArray

        NumClass = [0] * K
        # 初始化U
        p = 0
        for i in HiddenMatrix:
            for j in range(0, K):
                if i[m] == SetClass[j]:
                    NumClass[j] = NumClass[j] + 1
            p = p + 1

        train = []
        trainNum = 0
        test = []
        testNum = 0
        for i in range(0, K):
            train.append(int(NumClass[i] * 0.7))
            trainNum += int(NumClass[i] * 0.7)

            test.append(NumClass[i] - train[i])
            testNum += NumClass[i] - train[i]
        train_index = []
        test_index = []
        for i in range(0, K):
            #idx = np.random.choice(np.arange(NumClass[i]), size = train[i], replace = False)
            train_index.append(np.arange(train[i]))
            # print(train_index)
            # val_index.append(np.random.choice(np.delete(np.arange(NumClass[i]), train_index[i]), size = val[i], replace = False))
            test_index.append(np.arange(train[i],NumClass[i]))
            # print(test_index)
        dividX = []
        p2 = 0
        for i in range(0, K):
            dividX.append(HiddenMatrix[p2:p2 + NumClass[i], :])
            p2 = p2 + NumClass[i]

        trainSet = []
        for i in range(0, K):
            trainSet.append(dividX[i][train_index[i], :])
        TrainSet = np.zeros((0, m + 1))
        for i in range(0, K):
            TrainSet = np.vstack((TrainSet, trainSet[i]))
        # print(TrainSet)
        Y = TrainSet[:, m]
        # print(Y)
        TrainSet = np.delete(TrainSet, m, axis = 1)
        for i in range(0, K):
            trainSet[i] = np.delete(trainSet[i], m, axis = 1)

        testSet = []
        for i in range(0, K):
            testSet.append(np.delete(dividX[i][test_index[i], :], m, axis = 1))

        clf1 = GaussianNB()
        clf1.fit(TrainSet, Y)

        correct = 0
        for i in range(0, K):
            C = clf1.predict(testSet[i])
            # print(C)
            # print(SetClass[i])
            correct += sum(C == SetClass[i])
            # print(sum(C == SetClass[i]))
            # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
        testAcc = correct / testNum
        #arr_test.append(testacc)
        # print("test accuracy:{:.2%}".format(testacc))
        # print("---------------------------")
        correct1 = 0
        for i in range(0, K):
            C = clf1.predict(trainSet[i])
            # print(C)
            # print(SetClass[i])
            correct1 += sum(C == SetClass[i])
            # print(sum(C == SetClass[i]))
            # print("accuracy:{:.2%}".format((add + add1) / (val1+val2)))
        trainAcc=correct1 / trainNum
        return trainAcc,testAcc


    def test(self):
        X = np.loadtxt('../数据集/[013]segment(0-1).txt')

        m = X.shape[1] - 1  # 属性数量

        print(m)
        '''
        X = X[:, m - 160:m+1]
        m = X.shape[1] - 1
        '''

        n = X.shape[0]  # 样本数目 其中第一类1219，第二类683
        print(n)
        b = [9, 10, 11, 12, 16]
        vector_data = X[:, b]
        m=5
        # 提取label类别
        label_data = X[:, -1]
        cases = vector_data
        labels = vector_data
        Original_data=X
        '''
        cases = [
            [0.239362, 0.617571, 0.456522 ,0.536150 ,0.238095],
            [0.159574 ,0.728682, 0.646739 ,0.589736 ,0.208333],
            [0.239362 ,0.645995, 0.565217 ,0.516870 ,0.178571],
            [0.186170 ,0.609819, 0.565217 ,0.516019 ,0.238095],
            [0.212766 ,0.604651, 0.510870 ,0.520556 ,0.148810],
        ]
        labels = [[0.239362 ,0.617571, 0.456522 ,0.536150 ,0.238095],
                  [0.159574 ,0.728682, 0.646739 ,0.589736 ,0.208333],
                  [0.239362 ,0.645995, 0.565217 ,0.516870 ,0.178571],
                  [0.186170 ,0.609819, 0.565217 ,0.516019 ,0.238095],
                  [0.212766 ,0.604651, 0.510870 ,0.520556 ,0.148810]]
        '''
        Ytest1 = []
        Ytest2 = []
        Ytrain1 = []
        Ytrain2 = []
        X = []
        for p in range(1, 2):
            X.append(p)
            self.setup(m, p*m, m)
            Min_matrix = self.train(cases, labels, label_data, 500, 0.05, 0.1)
            Min_matrix = np.array(Min_matrix)
            print(Min_matrix)

            New_data = []
            for i in range(0, n):
                New_data.append(np.append(Min_matrix[i], label_data[i]))
            New_data = np.array(New_data)

            clf1 = GaussianNB()
            clf1.fit(New_data[:, :-1], New_data[:, -1])

            correct = 0
            C = clf1.predict(New_data[:, :-1])
            for i in range(0, n):
                if C[i] == label_data[i]:
                    correct = correct + 1
            print('neural-NBC ', correct / n)

            #acc1 = self.getAccurate(New_data)
            acc1=self.getTheAccurate(New_data)
            print("AE-NBC训练测试精度分别是", acc1[0], acc1[1])
            Ytrain1.append(acc1[0])
            Ytest1.append(acc1[1])

            ax1=plt.axes()
            sns.heatmap(pd.DataFrame(New_data[:, :-1]).corr(),ax = ax1)
            sns.pairplot(pd.DataFrame(New_data[:, :-1]))
            ax1.set_title("neural-NBC")
            plt.show()

            # ICA过滤无关数据集
            LargestAcc = 0
            New_data2 = []
            for j in range(m, 1, -1):
                transformer = FastICA(n_components = j, random_state = 0)
                X_transformed = transformer.fit_transform(vector_data)
                New_data1 = []
                for k in range(0, n):
                    New_data1.append(np.append(X_transformed[k], label_data[k]))
                New_data1 = np.array(New_data1)
                # print(X_transformed)
                # print(X_transformed.shape)

                clf2 = GaussianNB()
                clf2.fit(New_data1[:, :-1], New_data1[:, -1])

                correct = 0
                C1 = clf2.predict(New_data1[:, :-1])
                for t in range(0, n):
                    if C1[t] == label_data[t]:
                        correct = correct + 1
                if (correct / n) > LargestAcc:
                    LargestAcc = correct / n
                    New_data2 = New_data1
                # print('neural-ica-NBC ', correct / n)
            print('the largest neural-ica-NBC ', LargestAcc)

            #acc1 = self.getAccurate(New_data2)
            acc2=self.getTheAccurate(New_data2)
            print("AE-NBC-I训练测试精度分别是", acc2[0], acc2[1])
            Ytrain2.append(acc2[0])
            Ytest2.append(acc2[1])
            ax2 = plt.axes()
            sns.heatmap(pd.DataFrame(New_data1[:, :-1]).corr(),ax = ax2)
            sns.pairplot(pd.DataFrame(New_data2[:, :-1]))
            ax2.set_title("neural-ica-NBC")
            plt.show()

            # 测试NBC

            clf3 = GaussianNB()
            clf3.fit(vector_data, label_data)

            correct1 = 0
            C2 = clf3.predict(vector_data)
            for i in range(0, n):
                if C2[i] == label_data[i]:
                    correct1 = correct1 + 1
            print('NBC ', correct1 / n)

            acc1 = self.getTheAccurate(Original_data)
            print("NBC训练测试精度分别是", acc1[0], acc1[1])


            # ax3 = plt.axes()
            # sns.heatmap(pd.DataFrame(vector_data).corr(),ax=ax3)
            # sns.pairplot(pd.DataFrame(vector_data))
            # ax3.set_title("NBC")
            # plt.show()
            # New_data=np.vstack((Min_matrix,np.transpose(label_data)))
            # print(New_data)
            # print(get[0])
            # print(get[1])

        '''
        #plt.subplot(1,2,1)
        plt.plot(X, Ytest1, 's-', color = 'r', label = "test")  # s-:方形
        plt.plot(X, Ytrain1, 'o-', color = 'g', label = "train")  # o-:圆形
        plt.xlabel("the numbers of attribute")  # 横坐标名字
        plt.ylabel("accuracy")  # 纵坐标名字
        plt.title("AE-NBC")
        plt.show()

        #plt.subplot(1,2,2)
        plt.plot(X, Ytest2, 's-', color = 'r', label = "test")  # s-:方形
        plt.plot(X, Ytrain2, 'o-', color = 'g', label = "train")  # o-:圆形
        plt.xlabel("the numbers of attribute")  # 横坐标名字
        plt.ylabel("accuracy")  # 纵坐标名字
        plt.title("AE-NBC-I")
        plt.show()
        '''




if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
