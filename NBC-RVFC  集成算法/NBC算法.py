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

        self.smooth = 1  # 贝叶斯估计方法的平滑参数smooth=1，当smooth=0时即为最大似然估计
        self.p_prior = {}  # 先验概率
        self.p_condition = defaultdict(float)  # 条件概率
        self.AttributeIC = {}
        self.Wi = []
        self.AttributeW = defaultdict(float)
        self.AttributeY = defaultdict(float)
        self.realCondition = defaultdict(float)



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

    def predict_NBC(self, input_v):
        p_predict = {}
        # y为类别，p_y为每个类别的先验概率
        for y, p_y in self.p_prior.items():
            p = p_y  # 计算每种后验概率
            for d, v in enumerate(input_v):  # 0 2 1 0 0 2 1 0
                # print(d, v)
                p *= self.p_condition[(d, v, y)]
            p_predict[y] = p
        return p_predict




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



if __name__ == '__main__':

    X = np.loadtxt('../数据集/abalone(4,5).txt')

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
        k = 5
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
    arr_train_NBC=[]
    train_index_matrix=[[161, 118, 5, 102, 167, 174, 168, 74, 8, 181, 90, 58, 109, 139, 35, 78, 176, 128, 101, 119, 7, 46, 83, 56, 143, 136, 172, 18, 27, 156, 175, 80, 138, 116, 192, 84, 206, 81, 85, 154, 135, 96, 191, 57, 147, 145, 195, 110, 76, 89, 11, 2, 157, 55, 117, 134, 72, 71, 4, 201, 155, 113, 68, 22, 28, 153, 125, 9, 141, 207, 6, 61, 186, 107, 149, 93, 120, 17, 177, 15, 183, 124, 47, 160, 142, 26, 130, 132, 54, 98, 23, 3, 33, 179, 16, 30, 140, 38, 182, 14, 34, 165, 64, 133, 144, 162, 75, 196, 190, 40, 193, 36, 159, 67, 198, 187, 100, 148, 106, 53, 127, 48, 10, 92, 200, 1, 194, 66, 86, 21, 189, 173, 91, 178, 12, 60, 32, 164, 42, 105, 0, 25, 126, 97, 171], [104, 161, 115, 147, 30, 193, 38, 155, 49, 198, 191, 142, 172, 93, 98, 86, 45, 186, 200, 162, 154, 92, 178, 131, 160, 8, 18, 105, 76, 14, 22, 107, 67, 146, 163, 197, 7, 169, 164, 40, 29, 133, 165, 94, 69, 62, 137, 168, 166, 53, 21, 77, 176, 9, 122, 73, 17, 190, 120, 28, 91, 82, 174, 136, 4, 103, 78, 110, 152, 157, 112, 188, 202, 61, 111, 173, 66, 87, 42, 129, 34, 85, 150, 123, 180, 26, 2, 84, 100, 81, 41, 192, 64, 70, 130, 52, 125, 167, 187, 88, 23, 60, 96, 113, 75, 207, 44, 33, 175, 119, 47, 74, 114, 118, 36, 24, 135, 195, 13, 83, 149, 19, 37, 48, 194, 199, 127, 121, 138, 206, 128, 31, 185, 3, 1, 153, 134, 54, 124, 203, 55, 50, 126, 143, 140], [99, 124, 77, 20, 205, 195, 160, 175, 171, 162, 17, 155, 91, 204, 156, 181, 22, 119, 206, 38, 7, 83, 28, 201, 167, 157, 41, 61, 70, 194, 96, 84, 97, 198, 130, 62, 82, 161, 48, 147, 85, 123, 107, 105, 5, 50, 143, 134, 111, 100, 170, 81, 39, 190, 79, 112, 128, 24, 32, 138, 173, 37, 141, 176, 23, 2, 78, 80, 108, 69, 95, 191, 12, 98, 135, 51, 148, 73, 104, 172, 131, 64, 140, 102, 121, 163, 16, 164, 158, 187, 21, 45, 203, 149, 115, 86, 55, 31, 101, 43, 125, 193, 94, 184, 44, 33, 189, 113, 30, 59, 27, 207, 151, 188, 106, 117, 137, 54, 47, 144, 116, 154, 71, 58, 53, 14, 110, 174, 15, 72, 46, 142, 126, 11, 75, 200, 29, 13, 202, 19, 159, 132, 6, 103, 56], [125, 192, 106, 206, 119, 138, 46, 32, 174, 115, 76, 60, 200, 100, 199, 165, 0, 128, 203, 188, 61, 163, 51, 35, 84, 59, 50, 67, 63, 41, 185, 154, 173, 171, 205, 140, 92, 123, 198, 109, 201, 135, 10, 147, 90, 22, 118, 197, 28, 20, 110, 113, 72, 68, 6, 29, 58, 177, 146, 168, 62, 23, 25, 121, 182, 88, 64, 166, 153, 42, 156, 142, 8, 189, 1, 102, 101, 181, 37, 56, 175, 196, 77, 91, 4, 89, 87, 162, 73, 129, 53, 159, 83, 186, 31, 34, 134, 97, 131, 71, 112, 172, 145, 17, 107, 82, 38, 98, 7, 184, 167, 85, 204, 124, 57, 9, 75, 45, 19, 11, 139, 43, 122, 116, 180, 26, 183, 136, 160, 158, 3, 144, 86, 155, 170, 94, 55, 14, 151, 2, 54, 176, 40, 12, 103], [116, 89, 185, 44, 24, 207, 84, 85, 189, 187, 198, 25, 80, 148, 193, 49, 52, 7, 55, 50, 34, 125, 100, 156, 81, 123, 192, 21, 66, 35, 74, 32, 33, 131, 162, 154, 75, 98, 23, 203, 30, 179, 51, 42, 110, 134, 139, 145, 79, 177, 18, 180, 16, 58, 105, 199, 130, 144, 107, 143, 118, 161, 0, 106, 113, 171, 17, 166, 108, 93, 146, 64, 37, 186, 128, 175, 194, 14, 163, 5, 137, 122, 147, 10, 205, 1, 170, 56, 45, 141, 97, 43, 169, 60, 121, 94, 78, 120, 96, 47, 117, 138, 73, 86, 70, 102, 82, 99, 176, 181, 46, 68, 61, 67, 151, 159, 91, 62, 71, 126, 153, 36, 90, 22, 140, 6, 41, 3, 129, 124, 63, 167, 39, 188, 150, 190, 65, 95, 9, 20, 57, 158, 157, 178, 101], [159, 161, 5, 18, 177, 4, 61, 139, 102, 143, 41, 198, 156, 80, 75, 183, 146, 22, 172, 207, 202, 166, 173, 201, 154, 106, 1, 13, 167, 77, 195, 23, 88, 134, 179, 39, 48, 160, 20, 206, 132, 99, 135, 45, 115, 73, 117, 138, 25, 52, 123, 0, 89, 71, 116, 192, 204, 53, 196, 104, 125, 199, 171, 8, 19, 131, 108, 95, 68, 72, 174, 107, 164, 30, 103, 118, 194, 97, 7, 93, 114, 27, 70, 101, 62, 100, 140, 63, 197, 144, 12, 90, 186, 16, 10, 109, 17, 141, 60, 64, 32, 112, 170, 126, 187, 9, 54, 165, 128, 2, 119, 67, 55, 200, 69, 3, 142, 185, 162, 175, 29, 178, 59, 129, 43, 21, 136, 57, 50, 121, 188, 37, 190, 6, 133, 124, 28, 191, 205, 96, 85, 147, 163, 38, 58], [125, 110, 200, 150, 95, 121, 207, 56, 81, 144, 53, 206, 168, 80, 133, 184, 117, 101, 123, 132, 127, 92, 26, 103, 41, 77, 178, 84, 189, 62, 105, 114, 69, 157, 34, 122, 126, 63, 179, 204, 30, 42, 156, 196, 165, 100, 75, 87, 159, 28, 172, 108, 197, 181, 180, 137, 147, 166, 79, 71, 190, 18, 31, 3, 88, 38, 170, 116, 91, 5, 83, 19, 61, 23, 143, 124, 102, 162, 120, 93, 154, 58, 187, 74, 17, 107, 45, 160, 86, 49, 82, 55, 73, 6, 16, 185, 21, 111, 136, 199, 175, 47, 89, 35, 14, 186, 50, 39, 51, 57, 202, 152, 177, 7, 104, 9, 109, 97, 182, 163, 65, 198, 20, 64, 192, 54, 171, 195, 43, 129, 188, 68, 149, 4, 113, 85, 146, 155, 32, 106, 183, 37, 94, 48, 2], [127, 83, 190, 120, 20, 93, 11, 13, 5, 189, 95, 42, 69, 72, 117, 27, 70, 63, 145, 207, 102, 180, 198, 165, 150, 94, 50, 82, 59, 55, 193, 48, 177, 147, 104, 41, 36, 31, 201, 71, 163, 8, 35, 9, 113, 108, 76, 202, 75, 131, 67, 129, 110, 40, 116, 194, 92, 148, 44, 74, 121, 45, 149, 125, 105, 204, 98, 103, 128, 46, 23, 197, 68, 172, 175, 43, 26, 112, 188, 56, 73, 179, 119, 171, 140, 89, 159, 200, 122, 164, 22, 192, 191, 38, 79, 6, 19, 80, 186, 84, 88, 57, 155, 100, 91, 96, 90, 114, 49, 157, 168, 78, 87, 111, 99, 203, 3, 106, 133, 146, 124, 37, 123, 182, 28, 62, 109, 135, 173, 183, 134, 65, 25, 51, 166, 158, 184, 196, 14, 136, 16, 130, 29, 47, 12], [25, 46, 160, 166, 187, 104, 200, 44, 69, 115, 62, 143, 68, 168, 97, 35, 122, 23, 58, 201, 38, 121, 151, 86, 70, 56, 75, 164, 207, 130, 110, 60, 20, 31, 165, 52, 27, 206, 17, 33, 95, 99, 59, 50, 134, 55, 124, 48, 39, 182, 19, 142, 161, 180, 94, 77, 146, 153, 102, 203, 162, 144, 137, 149, 155, 78, 108, 76, 1, 82, 8, 133, 191, 74, 123, 120, 22, 131, 63, 53, 173, 18, 92, 172, 2, 81, 0, 88, 34, 136, 30, 125, 174, 11, 158, 154, 183, 202, 129, 5, 93, 61, 186, 128, 116, 3, 170, 114, 195, 87, 64, 147, 176, 126, 14, 140, 112, 132, 197, 49, 51, 101, 100, 83, 157, 205, 141, 45, 189, 138, 171, 26, 6, 177, 24, 119, 178, 89, 145, 43, 40, 80, 127, 167, 47], [201, 181, 37, 120, 102, 20, 154, 115, 72, 57, 70, 88, 172, 174, 91, 51, 162, 134, 189, 179, 188, 56, 99, 126, 200, 111, 71, 94, 28, 123, 159, 54, 122, 173, 138, 139, 130, 95, 55, 0, 2, 178, 164, 21, 151, 32, 113, 77, 165, 198, 161, 144, 135, 75, 191, 62, 205, 152, 59, 176, 25, 192, 106, 133, 44, 34, 171, 160, 150, 86, 40, 64, 157, 36, 116, 4, 52, 177, 74, 175, 109, 18, 98, 84, 156, 204, 50, 197, 83, 10, 78, 23, 193, 87, 145, 128, 85, 63, 170, 182, 7, 180, 96, 67, 110, 117, 76, 101, 155, 167, 203, 19, 168, 108, 184, 202, 45, 124, 38, 73, 27, 66, 158, 118, 190, 127, 186, 199, 22, 112, 60, 11, 93, 65, 30, 14, 89, 1, 147, 31, 125, 140, 3, 17, 33]]
    val_index_matrix=[[123, 199, 87, 115, 37, 112, 63, 20, 88, 151, 203, 39, 19, 104, 122, 51, 62, 94, 13, 29, 202, 163, 166, 169, 24, 103, 114, 111, 43, 146, 99, 204, 41, 129, 188, 170, 44, 152, 65, 73, 180, 95, 49, 70, 50, 82, 79, 31, 59, 197, 184, 108, 45, 69, 52, 150, 137, 77, 185, 205, 121, 158, 131], [99, 189, 183, 39, 145, 79, 97, 144, 171, 179, 205, 158, 116, 151, 90, 184, 80, 27, 63, 95, 32, 12, 6, 148, 56, 65, 43, 35, 71, 58, 11, 170, 59, 139, 108, 141, 89, 106, 10, 51, 16, 0, 101, 156, 5, 159, 20, 196, 182, 102, 201, 25, 15, 109, 117, 68, 181, 72, 204, 46, 57, 177, 132], [92, 26, 192, 153, 88, 196, 10, 179, 57, 146, 9, 133, 165, 87, 152, 36, 166, 42, 18, 182, 139, 60, 127, 114, 129, 52, 0, 63, 199, 136, 67, 4, 3, 183, 34, 89, 74, 66, 118, 150, 35, 185, 177, 180, 145, 8, 109, 168, 93, 40, 65, 76, 169, 197, 186, 1, 178, 49, 122, 120, 25, 68, 90], [49, 195, 65, 130, 161, 47, 157, 5, 137, 52, 120, 15, 33, 27, 141, 79, 178, 80, 149, 194, 74, 93, 108, 69, 18, 191, 30, 104, 114, 117, 24, 133, 193, 44, 81, 39, 48, 95, 13, 190, 150, 152, 78, 179, 143, 21, 207, 187, 169, 99, 111, 126, 132, 202, 127, 36, 164, 16, 148, 105, 70, 96, 66], [202, 168, 8, 87, 69, 40, 104, 4, 197, 11, 29, 13, 206, 155, 12, 15, 27, 88, 92, 31, 53, 165, 72, 127, 191, 112, 119, 109, 38, 164, 136, 174, 149, 114, 196, 77, 76, 135, 184, 152, 183, 172, 133, 103, 160, 111, 19, 115, 26, 195, 201, 132, 2, 200, 204, 142, 48, 182, 173, 59, 83, 54, 28], [203, 181, 92, 76, 122, 127, 51, 49, 151, 148, 168, 65, 180, 26, 66, 155, 176, 153, 11, 36, 81, 78, 158, 31, 86, 56, 98, 94, 42, 84, 184, 189, 120, 33, 137, 149, 14, 82, 169, 113, 40, 130, 83, 46, 35, 34, 15, 152, 74, 24, 157, 47, 193, 105, 79, 110, 182, 44, 145, 111, 91, 87, 150], [24, 131, 27, 12, 36, 151, 60, 15, 0, 1, 44, 10, 29, 33, 46, 135, 118, 98, 25, 66, 174, 13, 76, 194, 112, 203, 148, 70, 145, 119, 67, 134, 161, 173, 59, 130, 140, 96, 169, 72, 164, 115, 167, 22, 201, 138, 141, 52, 40, 139, 11, 78, 205, 142, 158, 193, 8, 176, 153, 191, 90, 99, 128], [206, 141, 167, 24, 176, 64, 39, 169, 10, 118, 174, 185, 195, 97, 81, 54, 21, 170, 162, 153, 60, 33, 2, 187, 138, 152, 86, 205, 53, 7, 154, 199, 178, 161, 30, 132, 1, 107, 151, 139, 126, 34, 4, 85, 18, 32, 181, 58, 17, 101, 61, 142, 77, 0, 144, 115, 66, 137, 160, 52, 156, 15, 143], [67, 90, 103, 21, 199, 41, 72, 15, 193, 204, 91, 190, 106, 198, 152, 4, 16, 163, 71, 139, 188, 36, 57, 175, 185, 113, 159, 184, 85, 13, 196, 37, 117, 109, 10, 111, 54, 169, 179, 7, 148, 65, 98, 105, 12, 192, 150, 32, 96, 28, 156, 118, 42, 79, 107, 29, 66, 194, 135, 84, 73, 9, 181], [146, 131, 194, 35, 81, 47, 100, 207, 104, 53, 119, 41, 46, 149, 92, 143, 9, 61, 141, 185, 6, 8, 114, 49, 29, 15, 16, 132, 80, 69, 187, 169, 166, 148, 105, 43, 42, 68, 5, 79, 12, 103, 107, 39, 195, 82, 48, 137, 90, 153, 13, 183, 196, 121, 26, 24, 136, 163, 58, 142, 129, 97, 206]]
    for k in range(0, 10):
        #train_index, val_index = prepare_data(X, 0.7)

        train_data = vector_data_NBC[train_index_matrix[k]]
        train_label = label_data_NBC[train_index_matrix[k]]

        test_data = vector_data_NBC[val_index_matrix[k]]
        test_label = label_data_NBC[val_index_matrix[k]]


        NBC_classific = NBC_RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
        NBC_classific.train_NBC(train_data, train_label)
        temp = 0
        correct = 0
        for x in test_data:

            # print(x)
            p_predict_sorted = sorted(NBC_classific.predict_NBC(x).items(), key = operator.itemgetter(1),
                                      reverse = True)
            # print(p_predict_sorted)
            # print(p_predict_sorted[0][0])
            if p_predict_sorted[0][0] == test_label[temp]:
                correct += 1
                #print(p_predict_sorted[0][0], test_label[temp])
            temp += 1
        print(correct / len(test_label))
        arr_test_NBC.append(correct / len(test_label))
        temp=0
        correct = 0
        for x in train_data:
            # print(x)
            p_predict_sorted = sorted(NBC_classific.predict_NBC(x).items(), key = operator.itemgetter(1),
                                      reverse = True)
            # print(p_predict_sorted)
            # print(p_predict_sorted[0][0])
            if p_predict_sorted[0][0] == train_label[temp]:
                correct += 1
                # print(p_predict_sorted[0][0], test_label[temp])
            temp += 1
        print(correct / len(train_label))
        arr_train_NBC.append(correct / len(train_label))


    arr_mean_test = np.mean(arr_test_NBC)
    arr_std_test = np.std(arr_test_NBC, ddof = 1)
    arr_mean_train=np.mean(arr_train_NBC)
    arr_std_train=np.std(arr_train_NBC)

    # arr_mean_NBC_RVFL=np.mean(arr_test_NBC_RVFL)
    # arr_std_NBC_RVFL=np.std(arr_test_NBC_RVFL,ddof = 1)
    print("测试集平均及标准差为", arr_mean_test, arr_std_test)
    print("训练集平均及标准差为",arr_mean_train,arr_std_train)
    # print("NBC-RVFL测试集平均及标准差为",arr_mean_NBC_RVFL,arr_std_NBC_RVFL)



