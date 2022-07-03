import numpy as np
import sklearn.datasets as sk_dataset
import random
import math
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
import sklearn.preprocessing as pre_processing
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer


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


# X = np.loadtxt('../数据集/[029]wineQW_small(0-1).txt')

X = np.loadtxt('../数据集1/optdigits.txt', delimiter = ',', dtype = np.str)
#X = np.loadtxt('../数据集/[018]musk01(0-1).txt')
# 缺失值补充

'''
X[X == '?'] = np.nan
imp=SimpleImputer(missing_values = np.nan,strategy = 'mean')
SimpleImputer(add_indicator = False,copy = True,fill_value = None,missing_values = '?',strategy = 'mean',verbose = 0)
imp.fit(X)
X=imp.transform(X)
'''



label = pre_processing.LabelEncoder()
X[:, -1] = label.fit_transform(X[:, -1])
print(X[:, -1])
X = X.astype(np.float)
print(X)

# print(X[:,13])
# X=np.delete(X,12,axis = 1)
# 归一化数据集

scaler = MinMaxScaler()
scaler.fit(X[:, :-1])
X[:, :-1] = scaler.transform(X[:, :-1])
print(X[:, -1])
print(X.dtype)
X[:, -1] = X[:, -1] + 1.0

m = X.shape[1] - 1
vector_data = X[:, :-1]
label_data = X[:, -1]

dict_label = Counter(X[:, -1])
K = len(dict_label)

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

vector_data_RVFL = X[:, :-1]
label_data_RVFL = X[:, -1]

train_accuracy_svc = []
test_accuracy_svc = []
test_accuracy_svc_PMSE=[]
test_accuracy_svc_AUC=[]

train_accuracy_NN = []
test_accuracy_NN = []
test_accuracy_NN_PMSE=[]
test_accuracy_NN_AUC=[]

train_accuracy_NB = []
test_accuracy_NB = []
test_accuracy_NB_PMSE=[]
test_accuracy_NB_AUC=[]

train_accuracy_cart = []
test_accuracy_cart = []
test_accuracy_cart_PMSE=[]
test_accuracy_cart_AUC=[]

train_accuracy_rt = []
test_accuracy_rt = []
test_accuracy_rt_PMSE=[]
test_accuracy_rt_AUC=[]

for k in range(0, 10):
    train_index, val_index = prepare_data(X, 0.7)
    train_data_NBC = vector_data_NBC[train_index]
    train_label_NBC = label_data_NBC[train_index]
    test_data_NBC = vector_data_NBC[val_index]
    test_label_NBC = label_data_NBC[val_index]

    train_data_RVFL = vector_data_RVFL[train_index]
    train_label_RVFL = label_data_RVFL[train_index]
    test_data_RVFL = vector_data_RVFL[val_index]
    test_label_RVFL = label_data_RVFL[val_index]

    # 支持向量机
    clf = GaussianNB()
    clf.fit(train_data_NBC, train_label_NBC)
    temp = 0
    correct = 0
    Predict_Matrix_test = clf.predict_proba(test_data_NBC)
    for class_num in range(Predict_Matrix_test.shape[0]):
        Predict_Matrix_test[class_num]=Predict_Matrix_test[class_num]/np.sum(Predict_Matrix_test[class_num])
    print(Predict_Matrix_test)
    result1 = np.argmax(Predict_Matrix_test, axis = 1)
    acc1 = np.sum(np.equal(result1, test_label_NBC - 1)) / len(test_label_NBC)
    test_accuracy_svc.append(acc1)

    PMSE = 0
    tag = 0
    for test_num in test_label_NBC:
        matrix = [0] * K
        matrix[int(test_num - 1)] = 1
        PMSE += np.sum(np.power(Predict_Matrix_test[tag] - matrix, 2))
        tag += 1
    test_accuracy_svc_PMSE.append(PMSE / len(test_label_NBC))
    # print(NBC_pro_matrix_test)
    # AUC排序性能比较
    # print(test_label)
    roc = metrics.roc_auc_score(test_label_NBC, Predict_Matrix_test, multi_class = 'ovo')
    test_accuracy_svc_AUC.append(roc)

    correct = 0
    Predict_Matrix_train = clf.predict(train_data_NBC)
    for i in range(len(train_label_RVFL)):
        if Predict_Matrix_train[i] == train_label_RVFL[i]:
            correct += 1
    train_accuracy_svc.append(correct / len(train_label_RVFL))
    print(test_accuracy_svc, train_accuracy_svc)

    # NN神经网路
    clf = MultinomialNB()
    clf.fit(train_data_NBC, train_label_NBC)
    correct = 0
    Predict_Matrix_test = clf.predict_proba(test_data_NBC)

    for class_num in range(Predict_Matrix_test.shape[0]):
        Predict_Matrix_test[class_num]=Predict_Matrix_test[class_num]/np.sum(Predict_Matrix_test[class_num])
    print(Predict_Matrix_test)

    result1 = np.argmax(Predict_Matrix_test, axis = 1)
    acc1 = np.sum(np.equal(result1, test_label_NBC - 1)) / len(test_label_NBC)
    test_accuracy_NN.append(acc1)

    PMSE = 0
    tag = 0
    for test_num in test_label_NBC:
        matrix = [0] * K
        matrix[int(test_num - 1)] = 1
        PMSE += np.sum(np.power(Predict_Matrix_test[tag] - matrix, 2))
        tag += 1
    test_accuracy_NN_PMSE.append(PMSE / len(test_label_NBC))
    # print(NBC_pro_matrix_test)
    # AUC排序性能比较
    # print(test_label)
    roc = metrics.roc_auc_score(test_label_NBC, Predict_Matrix_test, multi_class = 'ovo')
    test_accuracy_NN_AUC.append(roc)

    correct = 0
    Predict_Matrix_train = clf.predict(train_data_NBC)
    for i in range(len(train_label_RVFL)):
        if Predict_Matrix_train[i] == train_label_RVFL[i]:
            correct += 1
    train_accuracy_NN.append(correct / len(train_label_RVFL))
    print(test_accuracy_NN, train_accuracy_NN)

    '''
    #贝叶斯分类器
    clf=CategoricalNB()
    clf.fit(train_data_RVFL,train_label_RVFL)
    temp=0
    correct=0
    Predict_Matrix_test=clf.predict(test_data_RVFL)
    for i in range(len(test_label_RVFL)):
        if Predict_Matrix_test[i]==test_label_RVFL[i]:
            correct+=1
    test_accuracy_NB.append(correct/len(test_label_RVFL))

    correct=0
    Predict_Matrix_train=clf.predict(train_data_RVFL)
    for i in range(len(train_label_RVFL)):
        if Predict_Matrix_train[i]==train_label_RVFL[i]:
            correct+=1
    train_accuracy_NB.append(correct/len(train_label_RVFL))
    print(test_accuracy_NB,train_accuracy_NB)
    '''

    # 决策树C4.5
    clf = ComplementNB()
    clf.fit(train_data_NBC, train_label_NBC)
    temp = 0
    correct = 0
    Predict_Matrix_test = clf.predict_proba(test_data_NBC)
    for class_num in range(Predict_Matrix_test.shape[0]):
        Predict_Matrix_test[class_num] = Predict_Matrix_test[class_num] / np.sum(Predict_Matrix_test[class_num])
    print(Predict_Matrix_test)
    result1 = np.argmax(Predict_Matrix_test, axis = 1)
    acc1 = np.sum(np.equal(result1, test_label_NBC - 1)) / len(test_label_NBC)
    test_accuracy_cart.append(acc1)

    PMSE = 0
    tag = 0
    for test_num in test_label_NBC:
        matrix = [0] * K
        matrix[int(test_num - 1)] = 1
        PMSE += np.sum(np.power(Predict_Matrix_test[tag] - matrix, 2))
        tag += 1
    test_accuracy_cart_PMSE.append(PMSE / len(test_label_NBC))
    # print(NBC_pro_matrix_test)
    # AUC排序性能比较
    # print(test_label)
    roc = metrics.roc_auc_score(test_label_NBC, Predict_Matrix_test, multi_class = 'ovo')
    test_accuracy_cart_AUC.append(roc)

    correct = 0
    Predict_Matrix_train = clf.predict(train_data_NBC)
    for i in range(len(train_label_NBC)):
        if Predict_Matrix_train[i] == train_label_NBC[i]:
            correct += 1
    train_accuracy_cart.append(correct / len(train_label_NBC))
    print(test_accuracy_cart, train_accuracy_cart)

    # 随机森林
    clf = BernoulliNB()
    clf.fit(train_data_NBC, train_label_NBC)
    temp = 0
    correct = 0
    Predict_Matrix_test = clf.predict_proba(test_data_NBC)
    for class_num in range(Predict_Matrix_test.shape[0]):
        Predict_Matrix_test[class_num] = Predict_Matrix_test[class_num] / np.sum(Predict_Matrix_test[class_num])
    print(Predict_Matrix_test)
    result1 = np.argmax(Predict_Matrix_test, axis = 1)
    acc1 = np.sum(np.equal(result1, test_label_NBC - 1)) / len(test_label_NBC)
    test_accuracy_rt.append(acc1)

    PMSE = 0
    tag = 0
    for test_num in test_label_NBC:
        matrix = [0] * K
        matrix[int(test_num - 1)] = 1
        PMSE += np.sum(np.power(Predict_Matrix_test[tag] - matrix, 2))
        tag += 1
    test_accuracy_rt_PMSE.append(PMSE / len(test_label_NBC))
    # print(NBC_pro_matrix_test)
    # AUC排序性能比较
    # print(test_label)
    roc = metrics.roc_auc_score(test_label_NBC,Predict_Matrix_test, multi_class = 'ovo')
    test_accuracy_rt_AUC.append(roc)


    correct = 0
    Predict_Matrix_train = clf.predict(train_data_NBC)
    for i in range(len(train_label_NBC)):
        if Predict_Matrix_train[i] == train_label_NBC[i]:
            correct += 1
    train_accuracy_rt.append(correct / len(train_label_NBC))
    print(test_accuracy_rt, train_accuracy_rt)

arr_mean_train_svc = np.mean(train_accuracy_svc)
arr_std_train_svc = np.std(train_accuracy_svc)
arr_mean_test_svc = np.mean(test_accuracy_svc)
arr_std_test_svc = np.std(test_accuracy_svc)
arr_mean_test_svc_PMSE = np.mean(test_accuracy_svc_PMSE)
arr_std_test_svc_PMSE=np.std(test_accuracy_svc_PMSE)
arr_mean_test_svc_AUC = np.mean(test_accuracy_svc_AUC)
arr_std_test_svc_AUC = np.std(test_accuracy_svc_AUC)


print("FNBC测试集平均及标准差为", arr_mean_test_svc, arr_std_test_svc)
print("FNBC训练集平均值及标准差", arr_mean_train_svc, arr_std_train_svc)
print("FNBC_PMSE",arr_mean_test_svc_PMSE,arr_std_test_svc_PMSE)
print("FNBC_AUC",arr_mean_test_svc_AUC,arr_std_test_svc_AUC)

arr_mean_train_NN = np.mean(train_accuracy_NN)
arr_std_train_NN = np.std(train_accuracy_NN)
arr_mean_test_NN = np.mean(test_accuracy_NN)
arr_std_test_NN = np.std(test_accuracy_NN)
arr_mean_test_NN_PMSE = np.mean(test_accuracy_NN_PMSE)
arr_std_test_NN_PMSE=np.std(test_accuracy_NN_PMSE)
arr_mean_test_NN_AUC = np.mean(test_accuracy_NN_AUC)
arr_std_test_NN_AUC = np.std(test_accuracy_NN_AUC)

print("TAN测试集平均及标准差为", arr_mean_test_NN, arr_std_test_NN)
print("TAN训练集平均值及标准差", arr_mean_train_NN, arr_std_train_NN)
print("TAN_PMSE",arr_mean_test_NN_PMSE,arr_std_test_NN_PMSE)
print("TAN_AUC",arr_mean_test_NN_AUC,arr_std_test_NN_AUC)

arr_mean_train_cart = np.mean(train_accuracy_cart)
arr_std_train_cart = np.std(train_accuracy_cart)
arr_mean_test_cart = np.mean(test_accuracy_cart)
arr_std_test_cart = np.std(test_accuracy_cart)
arr_mean_test_cart_PMSE = np.mean(test_accuracy_cart_PMSE)
arr_std_test_cart_PMSE=np.std(test_accuracy_cart_PMSE)
arr_mean_test_cart_AUC = np.mean(test_accuracy_cart_AUC)
arr_std_test_cart_AUC = np.std(test_accuracy_cart_AUC)

print("AODE测试集平均及标准差为", arr_mean_test_cart, arr_std_test_cart)
print("AODE训练集平均值及标准差", arr_mean_train_cart, arr_std_train_cart)
print("AODE_PMSE",arr_mean_test_cart_PMSE,arr_std_test_cart_PMSE)
print("AODE_AUC",arr_mean_test_cart_AUC,arr_std_test_cart_AUC)

arr_mean_train_rt = np.mean(train_accuracy_rt)
arr_std_train_rt = np.std(train_accuracy_rt)
arr_mean_test_rt = np.mean(test_accuracy_rt)
arr_std_test_rt = np.std(test_accuracy_rt)
arr_mean_test_rt_PMSE = np.mean(test_accuracy_rt_PMSE)
arr_std_test_rt_PMSE=np.std(test_accuracy_rt_PMSE)
arr_mean_test_rt_AUC = np.mean(test_accuracy_rt_AUC)
arr_std_test_rt_AUC = np.std(test_accuracy_rt_AUC)

print("HNB测试集平均及标准差为", arr_mean_test_rt, arr_std_test_rt)
print("HNB训练集平均值及标准差", arr_mean_train_rt, arr_std_train_rt)
print("HNB_PMSE",arr_mean_test_rt_PMSE,arr_std_test_rt_PMSE)
print("HNB_AUC",arr_mean_test_rt_AUC,arr_std_test_rt_AUC)









