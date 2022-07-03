# -- coding: utf-8 --
# @Time : 2021/3/27 20:40
# @Author : HUANG XUYANG
# @Email : xhuang032@e.ntu.edu.sg
# @File : RVFL.py
# @Software: PyCharm

import numpy as np
import sklearn.datasets as sk_dataset
import random
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from collections import Counter, defaultdict
from minepy import MINE
import pandas as pd
import operator
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn import metrics

num_nides = 10  # Number of enhancement nodes.
regular_para = 1  # Regularization parameter.
weight_random_range = [-1, 1]  # Range of random weights.
bias_random_range = [0, 1]  # Range of random weights.


class NBC_RVFL:
    """A simple RVFL classifier.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
    """

    def __init__(self, n_nodes, lam, w_random_vec_range, b_random_vec_range, activation, same_feature = False):
        # RVFL初始化参数
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature
        # NBC初始化参数

        self.smooth = 1  # 贝叶斯估计方法的平滑参数smooth=1，当smooth=0时即为最大似然估计
        self.p_prior = {}  # 先验概率
        self.p_condition = defaultdict(float)  # 条件概率
        self.AttributeIC = {}
        self.Wi = []
        self.AttributeW = defaultdict(float)
        self.AttributeY = defaultdict(float)
        self.realCondition = defaultdict(float)

    def train_RVFL(self, data, label, n_class):
        """

        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        data = self.standardize(data)  # Normalization data
        n_sample = len(data)
        n_feature = len(data[0])
        self.random_weights = self.get_random_vectors(n_feature, self.n_nodes, self.w_random_range)
        self.random_bias = self.get_random_vectors(1, self.n_nodes, self.b_random_range)

        h = self.activation_function(
            np.dot(data, self.random_weights) + np.dot(np.ones([n_sample, 1]), self.random_bias))
        d = np.concatenate([h, data], axis = 1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis = 1)
        y = self.one_hot(label, n_class)
        if n_sample > (self.n_nodes + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)

    def predict_RVFL(self, data, output_prob = False):
        """

        :param data: Predict data.
        :param output_prob: A bool number, if True return the raw predict probability, if False return predict class.
        :return: Prediction result.
        """
        data = self.standardize(data)  # Normalization data
        h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        d = np.concatenate([h, data], axis = 1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis = 1)
        result = self.softmax(np.dot(d, self.beta))
        if not output_prob:
            result = np.argmax(result, axis = 1)
        return result

    def train_NBC(self, vector_data, label_data):

        n_samples = label_data.shape[0]  # 计算样本数
        # 统计不同类别的样本数，并存入字典，key为类别，value为样本数
        # Counter类的目的是用来跟踪值出现的次数。以字典的键值对形式存储，其中元素作为key，其计数作为value。
        dict_label = Counter(label_data)
        K = len(dict_label)
        # print(dict_label[-1])
        for key, val in dict_label.items():
            # 计算先验概率
            self.p_prior[key] = (val + self.smooth / K) / (n_samples + self.smooth)

        for dx in range(vector_data.shape[1]):
            # F(ai,c)
            nums_vd = defaultdict(int)
            # F(c)
            nums_vd1 = defaultdict(int)
            vector_dx = vector_data[:, dx]
            nums_sx = len(np.unique(vector_dx))
            for xd, y in zip(vector_dx, label_data):
                nums_vd[(xd, y)] += 1
                nums_vd1[(y)] += 1
            for key, val in nums_vd.items():
                self.p_condition[(dx, key[0], key[1])] = (val + self.smooth / nums_sx) / (
                        nums_vd1[(key[1])] + self.smooth)

    def predict_NBC(self, input_v, weight):
        p_predict = {}
        # y为类别，p_y为每个类别的先验概率
        for y, p_y in self.p_prior.items():
            p = p_y  # 计算每种后验概率
            for d, v in enumerate(input_v):  # 0 2 1 0 0 2 1 0
                # print(d, v)
                p *= self.p_condition[(d, v, y)] ** weight[d]
            p_predict[y] = p
        return p_predict

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: Accuracy.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        data = self.standardize(data)  # Normalization data
        h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        d = np.concatenate([h, data], axis = 1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis = 1)
        result = np.dot(d, self.beta)
        result = np.argmax(result, axis = 1)
        print(result)
        acc = np.sum(np.equal(result, label)) / len(label)
        return acc

    def get_random_vectors(self, m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    def one_hot(self, x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x):
        if self.same_feature is True:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x), 1 / np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x)
            return (x - self.data_mean) / self.data_std
        else:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x, axis = 0), 1 / np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x, axis = 0)
            return (x - self.data_mean) / self.data_std

    def softmax(self, x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis = 1))[:, np.newaxis], len(x[0]), axis = 1)


class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))

    def sine(self, x):
        return np.sin(x)

    def hardlim(self, x):
        return (np.sign(x) + 1) / 2

    def tribas(self, x):
        return np.maximum(1 - np.abs(x), 0)

    def radbas(self, x):
        return np.exp(-(x ** 2))

    def sign(self, x):
        return np.sign(x)

    def relu(self, x):
        return np.maximum(0, x)


def prepare_data(dataset, proportion):
    label = dataset[:, -1]
    data = dataset[:, :-1]

    n_class = len(set(label))

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    # print(train_index)
    val_index = shuffle_index[train_number:]
    # print(val_index)
    return train_index, val_index


def CountMutualXY(trainingSet):
    vector_data = trainingSet
    mine = MINE(alpha = 0.8, c = 15)
    MatrixI = np.zeros(shape = [vector_data.shape[1], vector_data.shape[1]])
    for dx in range(vector_data.shape[1]):
        for dy in range(vector_data.shape[1]):
            if dx != dy:
                vector_dx = vector_data[:, dx]
                vector_dy = vector_data[:, dy]
                mine.compute_score(vector_dx, vector_dy)
                MatrixI[dx, dy] = mine.mic()
    '''
    scaler = MinMaxScaler()
    scaler.fit(MatrixI)
    my_matrix_normorlize=scaler.transform(MatrixI)
    '''
    return MatrixI


def CountIndependenceMatrix(my_matrix):
    Average_R = 0
    # plt.plot(a,acc,color='green',label='NBC_RVFL')
    for x in range(my_matrix.shape[0]):
        for y in range(my_matrix.shape[1]):
            if x != y:
                Average_R += my_matrix[x][y]
            if x == y:
                my_matrix[x][y] = 1
    Average_R = Average_R / (my_matrix.shape[0] * (my_matrix.shape[0] - 1))
    # print("平均值",Average_R)

    # print(my_matrix.shape[0])

    add = my_matrix.sum(axis = 1)
    min_index = np.argmin(add)
    IndependentMatrix.append(min_index)
    min_index_1 = np.argmin(my_matrix[min_index])
    IndependentMatrix.append(min_index_1)
    # print(IndependentMatrix)

    Begin_I = my_matrix[min_index, min_index_1]
    for i in range(my_matrix.shape[1]):
        Min_I = 1
        Index_I = -1
        Add1 = 0
        for dy in range(my_matrix.shape[1]):
            if dy not in IndependentMatrix:
                IndependentMatrix.append(dy)
                combine = list(combinations(IndependentMatrix, 2))
                # print(combine)
                Add = 0
                for i in combine:
                    Add += my_matrix[i]
                # print(Add)
                # print(len(combine))
                Add = Add / len(combine)
                # print("单个Add",Add)
                if (Add - Begin_I) < Min_I and Add < Average_R:
                    Min_I = (Add - Begin_I)
                    Add1 = Add
                    Index_I = dy
                IndependentMatrix.pop()
        Begin_I = Add1
        print(Add1)
        # print(Begin_I)
        # print(Min_I)
        # print(Index_I)
        if Index_I != -1:
            IndependentMatrix.append(Index_I)
    for dy in range(my_matrix.shape[1]):
        if dy not in IndependentMatrix:
            IndependentMatrix.append(dy)
    # IndependentMatrix=IndependentMatrix[0:int(0.7*my_matrix.shape[1])]
    # print(IndependentMatrix[0:int(0.7*my_matrix.shape[1])])


def CountRelevantMatrix(my_matrix):
    for x in range(my_matrix.shape[0]):
        for y in range(my_matrix.shape[1]):
            if x == y:
                my_matrix[x][y] = 0
    print(my_matrix)
    add = my_matrix.sum(axis = 1)
    max_index = np.argmax(add)
    RelevantMatrix.append(max_index)
    max_index_1 = np.argmax(my_matrix[max_index])
    RelevantMatrix.append(max_index_1)
    # print(RelevantMatrix)
    Begin_I = my_matrix[max_index, max_index_1]
    for i in range(my_matrix.shape[1]):
        Max_I = 1
        Index_I = -1
        Add1 = 0
        for dy in range(my_matrix.shape[1]):
            if dy not in RelevantMatrix:
                RelevantMatrix.append(dy)
                combine = list(combinations(RelevantMatrix, 2))
                Add = 0
                for i in combine:
                    Add += my_matrix[i]
                # print(Add)
                # print(len(combine))
                Add = Add / len(combine)
                if (Begin_I - Add) < Max_I:
                    Max_I = (Begin_I - Add)
                    Add1 = Add
                    Index_I = dy
                RelevantMatrix.pop()
        Begin_I = Add1
        # print(Begin_I)
        # print(Min_I)
        # print(Index_I)
        if Index_I != -1:
            RelevantMatrix.append(Index_I)


arr_train = []
arr_test = []

if __name__ == '__main__':

    X = np.loadtxt('../数据集/[021]parkinsons(0-1).txt')

    '''
    sns.heatmap(abs(pd.DataFrame(X[:,:-1]).corr()), annot = True, vmin = 0, vmax = 1, cmap = "hot_r")
    sns.pairplot(pd.DataFrame(X[:,:-1]))
    plt.show()
    '''
    m = X.shape[1] - 1
    vector_data = X[:, :-1]
    label_data = X[:, -1]

    # 贝叶斯算法离散化后
    array1 = np.zeros(shape = (0, X.shape[0]))
    for n in range(0, m):
        k = 8
        d1 = pd.cut(vector_data[:, n], k, labels = range(k))
        array1 = np.vstack((array1, d1))
    array1 = np.vstack((array1, label_data))
    X1 = array1.T

    # print(X)
    # print(X1)
    vector_data_NBC = X1[:, :-1]
    label_data_NBC = X1[:, -1]

    # my_matrix = np.loadtxt("../数据集/[010]glass(0-1).txt")
    arr_test_NBC = []
    arr_test_NBC_1 = []
    arr_test_RVFL = []
    arr_test_NBC_RVFL = []

    Good_test_Accuracy=[]
    Good_train_Accuracy=[]
    length=0
    for k in range(30):

        train_index, val_index = prepare_data(X, 0.7)

        train_data = vector_data_NBC[train_index]
        train_label = label_data_NBC[train_index]

        test_data = vector_data_NBC[val_index]
        test_label = label_data_NBC[val_index]

        MatrixI = []
        IndependentMatrix = []
        RelevantMatrix = []

        getMatrixI = CountMutualXY(vector_data[train_index])
        CountIndependenceMatrix(getMatrixI)
        CountRelevantMatrix(getMatrixI)

        weight = defaultdict(float)
        Weight_NBC = []
        length=int(len(IndependentMatrix) * 0.6)
        '''
        for j in range(int(len(IndependentMatrix))):
            if j <= (int(len(IndependentMatrix) * 0.4)):
                weight[IndependentMatrix[j]] = 1 / (j + 1)
            else:
                weight[IndependentMatrix[j]] = 1 / (int(len(IndependentMatrix) * 0.4) + 1)

        
        '''

        for i in range(int(len(IndependentMatrix) * 0.6)):
            weight_temp = defaultdict(float)
            for j in range(int(len(IndependentMatrix))):
                weight_temp[IndependentMatrix[j]] = 1 / (i + 1)
            Weight_NBC.append(weight_temp)
        # print(Weight_NBC)

        for i in range(int(len(IndependentMatrix) * 0.6)):
            for j in range(i + 1):
                Weight_NBC[i][IndependentMatrix[j]] = 1 / (j + 1)

        # print(Weight_NBC)
        # 训练NBC
        NBC_classific = NBC_RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
        NBC_classific.train_NBC(train_data, train_label)

        test_accuracy = []
        for k in range(int(len(IndependentMatrix) * 0.6)):
            # print(num_class)
            # print(valSet)
            # IndependentMatrix=IndependentMatrix[0:int(0.8*getMatrixI.shape[1])]
            # 生成贝叶斯分类器独立组权重
            temp = 0
            correct = 0
            for x in test_data:
                print(NBC_classific.predict_NBC(x, Weight_NBC[k]))
                # print(x)
                p_predict_sorted = sorted(NBC_classific.predict_NBC(x, Weight_NBC[k]).items(),
                                          key = operator.itemgetter(1),
                                          reverse = True)
                # print(p_predict_sorted)
                # print(p_predict_sorted[0][0])
                if p_predict_sorted[0][0] == test_label[temp]:
                    correct += 1
                    # print(p_predict_sorted[0][0], test_label[temp])
                temp += 1
            print(correct / len(test_label))
            test_accuracy.append(correct / len(test_label))
        if test_accuracy[-1]>test_accuracy[0]:
           Good_test_Accuracy.append(test_accuracy)

        train_accuracy=[]
        for k in range(int(len(IndependentMatrix)*0.6)):

            temp = 0
            correct = 0
            for x in train_data:
                print(NBC_classific.predict_NBC(x, Weight_NBC[k]))
                # print(x)
                p_predict_sorted = sorted(NBC_classific.predict_NBC(x, Weight_NBC[k]).items(),
                                          key = operator.itemgetter(1),
                                          reverse = True)
                # print(p_predict_sorted)
                # print(p_predict_sorted[0][0])
                if p_predict_sorted[0][0] == train_label[temp]:
                    correct += 1
                    # print(p_predict_sorted[0][0], test_label[temp])
                temp += 1
            print(correct / len(train_label))
            train_accuracy.append(correct / len(train_label))

        if train_accuracy[-1]>train_accuracy[0]:
            Good_train_Accuracy.append(train_accuracy)




    Good_test_Accuracy=np.array(Good_test_Accuracy)

    ling_mean=np.mean(Good_test_Accuracy,axis = 0)
    ling_std1=np.mean(Good_test_Accuracy,axis = 0)+np.std(Good_test_Accuracy,axis = 0)
    ling_std2=np.mean(Good_test_Accuracy,axis = 0)-np.std(Good_test_Accuracy,axis = 0)
    print(length)
    print(np.std(Good_test_Accuracy,axis = 0))

    print(ling_mean)
    print(ling_std1)
    print(ling_std2)

    plt.plot(range(length),ling_mean,'-')
    plt.fill_between(range(length),ling_std1,ling_std2,alpha=0.3)
    #plt.ylim((0.5, 1))
    plt.xlabel('Weight Increment Times')
    plt.ylabel('Accuracy')
    plt.title('NBC_test')
    plt.show()

    Good_train_Accuracy = np.array(Good_train_Accuracy)

    ling_mean = np.mean(Good_train_Accuracy, axis = 0)
    ling_std1 = np.mean(Good_train_Accuracy, axis = 0) + np.std(Good_train_Accuracy, axis = 0)
    ling_std2 = np.mean(Good_train_Accuracy, axis = 0) - np.std(Good_train_Accuracy, axis = 0)
    print(length)
    print(np.std(Good_train_Accuracy, axis = 0))
    print(ling_mean)
    print(ling_std1)
    print(ling_std2)

    plt.plot(range(length), ling_mean, '-')
    plt.fill_between(range(length), ling_std1, ling_std2, alpha = 0.3)
    # plt.ylim((0.5, 1))
    plt.xlabel('Weight Increment Times')
    plt.ylabel('Accuracy')
    plt.title('NBC_train')
    plt.show()






    '''
    plt.plot(range(int(len(IndependentMatrix) * 0.6)), accuracy, color = 'r')
    plt.xlabel('number')
    plt.ylabel('accuracy')
    plt.show()
    
    '''


    arr_mean_NBC = np.mean(arr_test_NBC)
    arr_std_NBC = np.std(arr_test_NBC, ddof = 1)
    # arr_mean_NBC_RVFL=np.mean(arr_test_NBC_RVFL)
    # arr_std_NBC_RVFL=np.std(arr_test_NBC_RVFL,ddof = 1)
    print("NBC测试集平均及标准差为", arr_mean_NBC, arr_std_NBC)
    # print("NBC-RVFL测试集平均及标准差为",arr_mean_NBC_RVFL,arr_std_NBC_RVFL)




