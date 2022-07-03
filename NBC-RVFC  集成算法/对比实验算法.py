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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import  svm
from sklearn import tree

from sklearn.naive_bayes import GaussianNB

def prepare_data(dataset,proportion):
    label = dataset[:,-1]
    data = dataset[:,:-1]

    n_class = len(set(label))

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    #print(train_index)
    val_index = shuffle_index[train_number:]
    #print(val_index)
    return train_index,val_index



X = np.loadtxt('统计的数据集/整理好的数据集/adult(1).txt')



m=X.shape[1]-1
vector_data=X[:,:-1]
label_data=X[:,-1]

    # 贝叶斯算法离散化后
array1 = np.zeros(shape = (0, X.shape[0]))
for n in range(0, m):
    k = 8
    d1 = pd.cut(vector_data[:, n], k, labels = range(k))
    array1 = np.vstack((array1, d1))
array1 = np.vstack((array1, label_data))
X1 = array1.T

#print(X)
#print(X1)
vector_data_NBC = X1[:, :-1]
label_data_NBC = X1[:, -1]

vector_data_RVFL=X[:,:-1]
label_data_RVFL=X[:,-1]

train_accuracy_svc=[]
test_accuracy_svc=[]

train_accuracy_NN=[]
test_accuracy_NN=[]

train_accuracy_NB=[]
test_accuracy_NB=[]

train_accuracy_cart=[]
test_accuracy_cart=[]

train_accuracy_rt=[]
test_accuracy_rt=[]

for k in range(0, 1):
    train_index, val_index = prepare_data(X, 0.7)
    train_data_NBC = vector_data_NBC[train_index]
    train_label_NBC = label_data_NBC[train_index]
    test_data_NBC = vector_data_NBC[val_index]
    test_label_NBC = label_data_NBC[val_index]

    train_data_RVFL = vector_data_RVFL[train_index]
    train_label_RVFL = label_data_RVFL[train_index]
    test_data_RVFL = vector_data_RVFL[val_index]
    test_label_RVFL = label_data_RVFL[val_index]

    #支持向量机
    clf=svm.SVC()
    clf.fit(train_data_RVFL,train_label_RVFL)
    temp=0
    correct=0
    Predict_Matrix_test=clf.predict(test_data_RVFL)
    for i in range(len(test_label_RVFL)):
        if Predict_Matrix_test[i]==test_label_RVFL[i]:
            correct+=1
    test_accuracy_svc.append(correct/len(test_label_RVFL))

    correct=0
    Predict_Matrix_train=clf.predict(train_data_RVFL)
    for i in range(len(train_label_RVFL)):
        if Predict_Matrix_train[i]==train_label_RVFL[i]:
            correct+=1
    train_accuracy_svc.append(correct/len(train_label_RVFL))
    print(test_accuracy_svc,train_accuracy_svc)




    #NN神经网路
    clf = MLPClassifier(hidden_layer_sizes = (500,), activation = 'relu',
                       solver = 'lbfgs', alpha = 0.0001, batch_size = 'auto',
                       learning_rate = 'constant')
    clf.fit(train_data_RVFL,train_label_RVFL)

    temp=0
    correct=0
    Predict_Matrix_test=clf.predict(test_data_RVFL)
    for i in range(len(test_label_RVFL)):
        if Predict_Matrix_test[i]==test_label_RVFL[i]:
            correct+=1
    test_accuracy_NN.append(correct/len(test_label_RVFL))

    correct=0
    Predict_Matrix_train=clf.predict(train_data_RVFL)
    for i in range(len(train_label_RVFL)):
        if Predict_Matrix_train[i]==train_label_RVFL[i]:
            correct+=1
    train_accuracy_NN.append(correct/len(train_label_RVFL))
    print(test_accuracy_NN,train_accuracy_NN)




    #贝叶斯分类器

    clf = GaussianNB()

    Start = time.time()
    clf.fit(train_data_RVFL,train_label_RVFL)
    End = time.time()
    print("NB训练时间", End - Start)
    temp=0
    correct=0

    Start = time.time()
    Predict_Matrix_test=clf.predict(test_data_RVFL)


    for i in range(len(test_label_RVFL)):
        if Predict_Matrix_test[i]==test_label_RVFL[i]:
            correct+=1
    test_accuracy_NB.append(correct/len(test_label_RVFL))
    End = time.time()
    print("NB测试时间", End - Start)
    correct=0
    Predict_Matrix_train=clf.predict(train_data_RVFL)
    for i in range(len(train_label_RVFL)):
        if Predict_Matrix_train[i]==train_label_RVFL[i]:
            correct+=1
    train_accuracy_NB.append(correct/len(train_label_RVFL))
    print(test_accuracy_NB,train_accuracy_NB)


    


    #决策树C4.5
    clf=RandomForestClassifier()
    clf.fit(train_data_NBC,train_label_NBC)

    temp=0
    correct=0
    Predict_Matrix_test=clf.predict(test_data_NBC)
    for i in range(len(test_label_NBC)):
        if Predict_Matrix_test[i]==test_label_NBC[i]:
            correct+=1
    test_accuracy_cart.append(correct/len(test_label_NBC))

    correct=0
    Predict_Matrix_train=clf.predict(train_data_NBC)
    for i in range(len(train_label_NBC)):
        if Predict_Matrix_train[i]==train_label_NBC[i]:
            correct+=1
    train_accuracy_cart.append(correct/len(train_label_NBC))
    print(test_accuracy_cart,train_accuracy_cart)

    #随机森林
    clf=tree.DecisionTreeClassifier()
    clf.fit(train_data_NBC,train_label_NBC)
    temp=0
    correct=0
    Predict_Matrix_test=clf.predict(test_data_NBC)
    for i in range(len(test_label_NBC)):
        if Predict_Matrix_test[i]==test_label_NBC[i]:
            correct+=1
    test_accuracy_rt.append(correct/len(test_label_NBC))

    correct=0
    Predict_Matrix_train=clf.predict(train_data_NBC)
    for i in range(len(train_label_NBC)):
        if Predict_Matrix_train[i]==train_label_NBC[i]:
            correct+=1
    train_accuracy_rt.append(correct/len(train_label_NBC))
    print(test_accuracy_rt,train_accuracy_rt)



arr_mean_train_svc = np.mean(train_accuracy_svc)
arr_std_train_svc = np.std(train_accuracy_svc)
arr_mean_test_svc=np.mean(test_accuracy_svc)
arr_std_test_svc=np.std(test_accuracy_svc)

print("svc测试集平均及标准差为", arr_mean_test_svc, arr_std_test_svc)
print("svc训练集平均值及标准差", arr_mean_train_svc, arr_std_train_svc)

arr_mean_train_NN = np.mean(train_accuracy_NN)
arr_std_train_NN = np.std(train_accuracy_NN)
arr_mean_test_NN=np.mean(test_accuracy_NN)
arr_std_test_NN=np.std(test_accuracy_NN)

print("NN测试集平均及标准差为", arr_mean_test_NN, arr_std_test_NN)
print("NN训练集平均值及标准差", arr_mean_train_NN, arr_std_train_NN)

arr_mean_train_NB = np.mean(train_accuracy_NB)
arr_std_train_NB= np.std(train_accuracy_NB)
arr_mean_test_NB=np.mean(test_accuracy_NB)
arr_std_test_NB=np.std(test_accuracy_NB)

print("NB测试集平均及标准差为", arr_mean_test_NB, arr_std_test_NB)
print("NB训练集平均值及标准差", arr_mean_train_NB, arr_std_train_NB)

arr_mean_train_cart = np.mean(train_accuracy_cart)
arr_std_train_cart= np.std(train_accuracy_cart)
arr_mean_test_cart=np.mean(test_accuracy_cart)
arr_std_test_cart=np.std(test_accuracy_cart)

print("cart测试集平均及标准差为", arr_mean_test_cart, arr_std_test_cart)
print("cart训练集平均值及标准差", arr_mean_train_cart, arr_std_train_cart)

arr_mean_train_rt = np.mean(train_accuracy_rt)
arr_std_train_rt= np.std(train_accuracy_rt)
arr_mean_test_rt=np.mean(test_accuracy_rt)
arr_std_test_rt=np.std(test_accuracy_rt)

print("rt测试集平均及标准差为", arr_mean_test_rt, arr_std_test_rt)
print("rt训练集平均值及标准差", arr_mean_train_rt, arr_std_train_rt)









