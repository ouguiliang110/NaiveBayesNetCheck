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
import copy

num_nides = 10  # Number of enhancement nodes.
regular_para = 1  # Regularization parameter.
weight_random_range = [-1, 1]  # Range of random weights.
bias_random_range = [0, 1]  # Range of random weights.


class RVFL:
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

    def train(self, data, label, n_class):
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
        return self.random_weights,self.random_bias
    def train_weight(self,data,label,n_class,random_weight,random_bias):
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
        self.random_weights = random_weight
        self.random_bias = random_bias

        h = self.activation_function(
            np.dot(data, self.random_weights) + np.dot(np.ones([n_sample, 1]), self.random_bias))
        d = np.concatenate([h, data], axis = 1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis = 1)
        y = self.one_hot(label, n_class)
        if n_sample > (self.n_nodes + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)

    def predict(self, data, output_prob = False):
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
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


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
                for j in combine:
                    Add += my_matrix[j]
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
        if Index_I != -1 and i >= 2:
            IndependentMatrix.append(Index_I)
    # IndependentMatrix=IndependentMatrix[0:int(0.7*my_matrix.shape[1])]
    # print(IndependentMatrix[0:int(0.7*my_matrix.shape[1])])


def CountRelevantMatrix(my_matrix):
    for x in range(my_matrix.shape[0]):
        for y in range(my_matrix.shape[1]):
            if x == y:
                my_matrix[x][y] = 0
    # print(my_matrix)
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

    X = np.loadtxt('../数据集/[014]ionosphere(0-1).txt')
    '''
    sns.heatmap(abs(pd.DataFrame(X[:,:-1]).corr()), annot = True, vmin = 0, vmax = 1, cmap = "hot_r")
    sns.pairplot(pd.DataFrame(X[:,:-1]))
    plt.show()X
    '''
    label = X[:, -1]
    data = X[:, :-1]

    n_class = len(set(label))

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(0.7 * len(label))
    train_index = shuffle_index[:train_number]
    # print(train_index)
    val_index = shuffle_index[train_number:]
    # print(val_index)
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data
    label_val = label

    # my_matrix = np.loadtxt("../数据集/[010]glass(0-1).txt")
    arr_test_NBC = []
    arr_test_NBC_1 = []
    arr_test_RVFL = []
    arr_test_NBC_RVFL = []

    MatrixI = []
    IndependentMatrix = []
    RelevantMatrix = []

    getMatrixI = CountMutualXY(data_train)
    CountIndependenceMatrix(getMatrixI)
    CountRelevantMatrix(getMatrixI)
    print(RelevantMatrix)
    #生成一个矩阵相乘数据集
    a=np.ones((int(len(RelevantMatrix)*0.6),len(RelevantMatrix)))
    for i in range(len(RelevantMatrix)):
        a[:, i] = 1 / (i + 1)
    for i in range(int(len(RelevantMatrix)*0.6)):
        for j in range(i+1,len(RelevantMatrix)):
            a[i][j]=1/(i+1)

    accuracy= []

    correct_matrix={}
    error_matrix={}

    for i in range(int(len(RelevantMatrix)*0.6)):
        #train, test, num_class = prepare_data(X, 0.7)

        # print(num_class)
        # print(valSet)
        min_train=copy.deepcopy(data_train)
        min_val=copy.deepcopy(data_val)


        #print(a)
        # a = np.linspace(0, 9, len(RelevantMatrix), base = 1 / 2)
        # print(IndependentMatrix)
        j=0
        for k in RelevantMatrix:
            min_train[:, k - 1] = min_train[:, k - 1] * a[i][j]
            min_val[:, k - 1] = min_val[:,  k - 1] * a[i][j]
            j+=1



        '''
        IndependentMatrix=IndependentMatrix[0:int(0.6*getMatrixI.shape[1])]
        RelevantMatrix = RelevantMatrix[0:int(0.6* getMatrixI.shape[1])]
        #print(IndependentMatrix)
        #print(RelevantMatrix)
        #print(train[0])
        #print(train[1])
        NBTrainingMatrix=train[0][:,IndependentMatrix]
        RVFLTrainingMatrix=train[0][:,RelevantMatrix]
        NBTestingMatrix=test[0][:,IndependentMatrix]
        RVFLTestingMatrix=test[0][:,RelevantMatrix]

        '''

        rvfl1 = RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
        B = label_train.astype(int)
        '''
        # nb准确率
        clf = GaussianNB()
        clf.fit(train[0], train[1] - 1)
        NB_Pro = clf.predict_proba(test[0])
        result2 = np.argmax(NB_Pro, axis = 1)
        acc2 = np.sum(np.equal(result2, test[1] - 1)) / len(test[1])
        arr_test_NBC.append(acc2)
        '''

        '''
        clf1 = GaussianNB()
        clf1.fit(NBTrainingMatrix, train[1] - 1)
        NB_Pro = clf1 .predict_proba(NBTestingMatrix)
        result3 = np.argmax(NB_Pro, axis = 1)
        acc3 = np.sum(np.equal(result3, test[1]- 1)) / len(test[1])
        arr_test_NBC_1.append(acc3)
        '''
        if i==0:
            random=rvfl1.train(min_train, B - 1, n_class)
        else:
            rvfl1.train_weight(min_train,B-1,n_class,random[0],random[1])

        # rvfl和nb集成准确率

        RVFLProbability = rvfl1.predict(min_val, output_prob = True)
        result3 = np.argmax(RVFLProbability, axis = 1)
        for i in range(len(result3)):
            if result3[i]==(label_val[i]-1):
                correct_matrix[i]=RVFLProbability[i]
            else:
                error_matrix[i] = RVFLProbability[i]

        acc2 = np.sum(np.equal(result3, label_val - 1)) / len(label_val)
        print(acc2)
        accuracy.append(acc2)
        # print(RVFLProbability)
        '''
        clf=GaussianNB()
        clf.fit(NBTrainingMatrix,train[1]-1)
        NBProbability=clf.predict_proba(NBTestingMatrix)
        #print(NBProbability)
        #Lamuda=
        ADD_Probability=RVFLProbability+NBProbability
        #print(ADD_Probability)
        result3 = np.argmax(ADD_Probability, axis = 1)
        #print(result)
        acc3 = np.sum(np.equal(result3, test[1]-1)) / len(test[1])
        arr_test_NBC_RVFL.append(acc3)
        print(acc3)
        '''

    print(arr_test_NBC)
    plt.plot(range(int(len(RelevantMatrix)*0.6)),accuracy,color='r')
    plt.xlabel('number')
    plt.ylabel('accuracy')
    plt.show()

    '''
    arr_mean_NBC = np.mean(arr_test_NBC)
    arr_std_NBC = np.std(arr_test_NBC, ddof = 1)
    arr_mean_NBC_1 = np.mean(arr_test_NBC_1)
    arr_std_NBC_1 = np.std(arr_test_NBC_1, ddof = 1)
    print("NBC测试集平均及标准差为", arr_mean_NBC, arr_std_NBC)
    print("NBC-RVFL测试集平均及标准差为", arr_mean_NBC_1, arr_std_NBC_1)
    '''


'''
    train, val, num_class = prepare_data(0.8)
    rvfl = RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
    rvfl.train(train[0], train[1], num_class)
    prediction = rvfl.predict(val[0], output_prob=True)
    print(prediction)
    accuracy = rvfl.eval(val[0], val[1])
    print(accuracy)

'''


