#-- coding: utf-8 --
#@Time : 2021/3/27 20:40
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : RVFL.py
#@Software: PyCharm


import numpy as np
import sklearn.datasets as sk_dataset
import pandas as pd
import  sklearn.preprocessing as pre_processing
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random

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
    def __init__(self, n_nodes, lam, w_random_vec_range, b_random_vec_range, activation, same_feature=False):
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

        h = self.activation_function(np.dot(data, self.random_weights) + np.dot(np.ones([n_sample, 1]), self.random_bias))
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        y = self.one_hot(label, n_class)
        if n_sample > (self.n_nodes + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)

    def predict(self, data, output_prob=False):
        """

        :param data: Predict data.
        :param output_prob: A bool number, if True return the raw predict probability, if False return predict class.
        :return: Prediction result.
        """
        data = self.standardize(data)  # Normalization data
        h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        result = self.softmax(np.dot(d, self.beta))
        if not output_prob:
            result = np.argmax(result, axis=1)
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
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        result = np.dot(d, self.beta)
        result = np.argmax(result, axis=1)
        print(result)
        acc = np.sum(np.equal(result, label))/len(label)
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
                self.data_std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x)
            return (x - self.data_mean) / self.data_std
        else:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x, axis=0)
            return (x - self.data_mean) / self.data_std


    def softmax(self, x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)


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
        return np.exp(-(x**2))

    def sign(self, x):
        return np.sign(x)

    def relu(self, x):
        return np.maximum(0, x)


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
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class,train_index,val_index

def decriDateset(self, dataset):
    array = np.zeros(shape = (0, dataset.shape[1]))
    for i in dataset:
        k = 5
        d1 = pd.cut(i, k, labels = range(k))
        array = np.vstack((array, d1))
    # 以《统计学习方法》中的例4.1计算，为方便计算，将例子中"S"设为0，“M"设为1。
    # 提取特征向量
    return array
# 主要求E的值
def getE(l, j):
    E = 0
    for i in range(0, n):
        E = E + U[i][l] * (X[i][j] - Z[l][j]) ** 2
    E = E + n1
    return E


# 求GY
def getGY(j, l):
    add2 = 0
    for t in range(0, T):
        add2 = add2 + G[j][t] * (Y[l][t] ** 2)
    return add2


# 求GYT
def getGYV(j, l):
    add1 = 0
    for t in range(0, T):
        add1 = add1 + G[j][t] * (Y[l][t] ** 2) * V[l][t]
    return add1
#求其中F

def getF(j, t):
    add8 = 0
    for l in range(0, K):
        add8 += (Y[l, t] ** 2) * (W[l, j] - V[l, t]) ** 2
    return add8
#求其中H

def getH(l, t):
    add8 = 0
    for j in range(0, m):
        add8 += G[j, t] * (W[l, j] - V[l, t]) ** 2
    return add8 + n2

def getRandom(num):
    Ran = np.random.dirichlet(np.ones(num), size = 1)
    Ran = Ran.flatten()
    return Ran


if __name__ == '__main__':
    #X=pd.read_csv('../数据集/HCV-Egy-Data.csv')
    #X=np.array(X)
    #print(X)
    X = np.loadtxt('../数据集1/contraceptive.txt',delimiter = ',',dtype = np.str)
    #X = np.loadtxt('../数据集/contraceptive.txt')
    #缺失值补充
    '''
    X[X == '?'] = np.nan
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    SimpleImputer(add_indicator = False, copy = True, fill_value = None, missing_values = '?', strategy = 'mean',
                  verbose = 0)
    imp.fit(X)
    X = imp.transform(X)
    '''


    label = pre_processing.LabelEncoder()
    X[:, -1] = label.fit_transform(X[:, -1])
    print(X[:, -1])
    X = X.astype(np.float)
    print(X)

    #print(X[:,13])
    #X=np.delete(X,12,axis = 1)
    #归一化数据集

    scaler = MinMaxScaler()
    scaler.fit(X[:,:-1])
    X[:,:-1]= scaler.transform(X[:,:-1])
    print(X[:,-1])

    print(X.dtype)

    X[:,-1]=X[:,-1]+1.0
    #X=X[0:1500,:]
    #X[X==7]=6
    print(X)
    print(X.dtype)


    # 主要的参数设置
    n1 = 0.0001
    n2 = 0.0001
    n3 = 0.00001
    N = 100
    b = 1  # β值大小
    T = 2 # 组数量大小
    m = X.shape[1] - 1  # 属性数量
    n = X.shape[0]  # 样本数目 其中第一类212，第二类55
    print(X.shape)

    SetClass = set(X[:, m])
    SetClass = list(map(int, SetClass))
    print(SetClass)
    K = len(SetClass)  # 类标 记数量

    Z = np.zeros((K, m))
    U = np.zeros((n, K))

    # print(X.shape)
    # print(U)
    # print(X)

    # 创建空数组

    # 处理数据集20是根据最大类数来处理得到的
    newarray = [np.zeros(shape = [0, m + 1])] * 30
    for i in X:
        for j in SetClass:
            if i[m] == j:
                newarray[j] = np.vstack((newarray[j], i))

    NewArray = np.zeros(shape = [0, m + 1])
    for i in SetClass:
        NewArray = np.vstack((NewArray, newarray[i]))
    #np.savetxt('../数据集/2satimage.txt', NewArray)
    print(NewArray)
    print(NewArray.shape)
    print(newarray)

    # 统计各类数量
    NumClass = [0] * K
    # 初始化U
    p = 0
    for i in X:
        for j in range(0, K):
            if i[m] == SetClass[j]:
                U[p][j - 1] = 1
                NumClass[j] = NumClass[j] + 1
        p = p + 1
    print(NumClass)

    X = np.delete(NewArray, m, axis = 1)

    p = 0
    for i in SetClass:
        temp = np.delete(newarray[i], m, axis = 1)
        Z[p] = np.mean(temp, axis = 0)
        # print(Z[p])
        p = p + 1

    # 初始化类属性W，属性分组矩阵y，类组矩阵V为1
    W = np.ones((K, m))
    V = np.ones((K, T))
    Y = np.ones((K, T))

    # 初始化G
    G = np.zeros((m, T))

    G[:, 0] = 1
    # print(G)
    # 初始化s,Q
    '''标记循环次数的主要参数'''
    s = 0
    '''目标函数的大小'''
    Q = 0
    '''循环次数'''
    goNum = 0

    # 主要的循环过程
    # 由算法可知类标记已经确定，则Z和U无需更新，只需要更新W
    while True:
        # 求W 其中最难最复杂的内容
        for l in range(0, K):
            for j in range(0, m):
                # 求h1的值
                add1 = getGYV(j, l)
                h1 = b * add1
                # 求h2的值
                add2 = getGY(j, l)
                add2 = b * add2
                E1 = getE(l, j)
                h2 = add2 + E1
                # 求h3的值
                add4 = 0
                for h in range(0, m):
                    num1 = getGY(h, l)
                    num1 = num1 * b
                    E2 = getE(l, h)
                    num2 = getGYV(h, l)
                    num2 = num2 * b
                    add4 = add4 + (1 / (num1 + E2)) * num2
                h3 = add4 - m
                # 求h4的值
                add5 = 0
                for h in range(0, m):
                    num3 = b * getGY(h, l)
                    E3 = getE(l, h)
                    add5 = add5 + 1 / (num3 + E3)
                h4 = (b * getGY(j, l) + getE(l, j)) * add5
                W[l][j] = h1 / h2 - h3 / h4
        # 主要条件判断更新V矩阵
        if s == 0:
            # 初始化V1
            for t in range(0, T):
                ranNum = random.randint(0, m - 1)
                V[:, t] = W[:, ranNum]
        else:
            add6 = 0
            add7 = 0
            for l in range(0, K):
                for t in range(0, T):
                    for j in range(0, m):
                        add6 = add6 + G[j, t] * W[l, j]
                        add7 = add7 + G[j, t]
                    V[l, t] = add6 / add7
        # 更新G矩阵
        for j in range(0, m):
            for t in range(0, T):
                Ft = getF(j, t)
                flag = 1
                for s in range(0, T):
                    if Ft > getF(j, s):
                        flag = 0
                G[j, t] = flag

        # 更新Y矩阵
        for l in range(0, K):
            for t in range(0, T):
                add9 = 0
                for s in range(0, K):
                    add9 += getH(l, t) / getH(s, t)
                Y[l, t] = 2 / add9
        # 主要目标函数进行求解
        s = s + 1
        Qnum1 = 0
        for l in range(0, K):
            for i in range(0, n):
                add10 = 0
                for j in range(0, m):
                    add10 += (W[l, j] ** 2) * ((X[i, j] - Z[l, j]) ** 2)
                Qnum1 += U[i, l] * add10
        Qnum2 = 0
        for l in range(0, K):
            for j in range(0, m):
                Qnum2 += W[l, j] ** 2
        Qnum2 = Qnum2 * n1
        Qnum3 = 0
        for t in range(0, T):
            for j in range(0, m):
                add11 = 0
                for l in range(0, K):
                    add11 += (Y[l, t] ** 2) * ((W[l, j] - V[l, t]) ** 2)
                Qnum3 += G[j, t] * add11
        Qnum3 = Qnum3 * b
        Qnum4 = 0
        for l in range(0, K):
            for t in range(0, T):
                Qnum4 += Y[l, t] ** 2
        Qnum4 = n2 * Qnum4 * b
        Q1 = Qnum1 + Qnum2 + Qnum3 + Qnum4
        if abs(Q1 - Q) < n3 or s >= N:
            break
        Q = Q1
        goNum += 1
        print(goNum)
        print(np.sum(G, axis = 0))
    print(W)
    # print(np.sum(W,axis = 1))
    print("---------------------")
    print(G)
    print(np.sum(G, axis = 0))
    print("----------------------")

    Group = []
    for t in range(0, T):
        print("第" + repr(t) + "组")
        list1 = []
        for j in range(0, m):
            if G[j, t] == 1:
                list1.append(j)
            else:
                continue
        Group.append(list1)
        print(list1)
        print('\n')
    print("---------------------")
    for i in range(0, len(Group)):
        print(Group[i])


    arr_test_RVFL = []
    arr_training_RVFL=[]
    X=NewArray

    '''
    计算p(m)
    '''
    dict_label = Counter(X[:,-1])
    K = len(dict_label)
    # print(dict_label[-1])
    p_prior = {}  # 先验概率
    smooth=1
    for key, val in dict_label.items():
        # 计算先验概率
        p_prior[key] = (val + smooth / K) / (X.shape[0] + smooth)
    train_index_matrix=[]
    val_index_matrix=[]
    roc_matrix=[]
    PMSE_matrix=[]
    for k in range(0,1):
        train, test, num_class, train_index, val_index = prepare_data(X, 0.7)
        train_index_matrix.append(list(train_index))
        val_index_matrix.append(list(val_index))
        Rvfl_probablity_matrix=np.ones((test[0].shape[0], num_class))
        Rvfl_probablity_matrix_train = np.ones((train[0].shape[0], num_class))


        for t in range(T):
            if Group[t]:
                Rvf_one = RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
                Rvf_two = RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
                B = train[1].astype(int)
                Rvf_one.train(train[0][:, Group[t]], B - 1, num_class)
                Rvf_two.train(train[0][:, Group[t]], B - 1, num_class)
                Rvfl_Pro_one = Rvf_one.predict(test[0][:, Group[t]], output_prob = True)
                Rvfl_Pro_two=Rvf_two.predict(train[0][:, Group[t]],output_prob = True)
                Rvfl_probablity_matrix *= Rvfl_Pro_one
                Rvfl_probablity_matrix_train*=Rvfl_Pro_two
            else:
                continue
        for class_num in range(test[0].shape[0]):
            Rvfl_probablity_matrix[class_num]=Rvfl_probablity_matrix[class_num]/np.sum(Rvfl_probablity_matrix[class_num])
            Rvfl_probablity_matrix_train[class_num] = Rvfl_probablity_matrix_train[class_num] / np.sum(
                Rvfl_probablity_matrix_train[class_num])

        for class_num in range(K):
            Rvfl_probablity_matrix[class_num]=Rvfl_probablity_matrix[class_num]*(1/p_prior[class_num+1])**2
            Rvfl_probablity_matrix_train[class_num] = Rvfl_probablity_matrix_train[class_num] * (1 / p_prior[class_num + 1]) ** 2

        for class_num in range(test[0].shape[0]):
            Rvfl_probablity_matrix[class_num] = Rvfl_probablity_matrix[class_num] / np.sum(
                Rvfl_probablity_matrix[class_num])
            Rvfl_probablity_matrix_train[class_num] = Rvfl_probablity_matrix_train[class_num]/np.sum(Rvfl_probablity_matrix_train[class_num])
        #print(Rvfl_probablity_matrix)

        for i in range(K):
            print('均值和标准差', np.mean(Rvfl_probablity_matrix[:, i]), np.std(Rvfl_probablity_matrix[:, i]))
            x=random.normal(loc=np.mean(Rvfl_probablity_matrix[:,i]),scale=np.std(Rvfl_probablity_matrix[:,i]),size=50000)
            sns.kdeplot(x,bw = 0.05,shade = True,label = "Class"+str(i+1))
        plt.legend()
        plt.show()

        '''
        sns.kdeplot(Rvfl_probablity_matrix[:, 2], bw = 0.25, shade = True, label = "Class1")
        sns.kdeplot(Rvfl_probablity_matrix[:, 3], bw = 0.25, shade = True, label = "Class2")
        sns.kdeplot(Rvfl_probablity_matrix[:, 4], bw = 0.25, shade = True, label = "Class1")
        sns.kdeplot(Rvfl_probablity_matrix[:, 5], bw = 0.25, shade = True, label = "Class2")
        '''

        #sns.distplot(Rvfl_probablity_matrix[:, 2], label = "Class3")

        #生成类条件概率估计质量的比较值
        PMSE=0
        tag=0
        for test_num in test[1]:
            matrix=[0]*K
            matrix[int(test_num-1)]=1
            PMSE+=np.sum(np.power(Rvfl_probablity_matrix[tag]-matrix,2))
            tag+=1
        PMSE_matrix.append(PMSE/len(test[1]))

        result1 = np.argmax(Rvfl_probablity_matrix, axis = 1)
        acc1 = np.sum(np.equal(result1, test[1] - 1)) / len(test[1])
        arr_test_RVFL.append(acc1)

        result2=np.argmax(Rvfl_probablity_matrix_train,axis = 1)
        acc2=np.sum(np.equal(result2,train[1]-1))/len(train[1])
        arr_training_RVFL.append(acc2)

        #AUC排序性能比较
        print(test[1])
        #print(Rvfl_probablity_matrix)
        '''
        roc= metrics.roc_auc_score(test[1],Rvfl_probablity_matrix[:,1],multi_class = 'ovo')
        roc_matrix.append(roc)
        '''




    print('-------------------')
    print(train_index_matrix)
    print('---------------------')
    print(val_index_matrix)

    arr_mean_RVFL = np.mean(arr_test_RVFL)
    arr_std_RVFL = np.std(arr_test_RVFL)
    arr_mean_train=np.mean(arr_training_RVFL)
    arr_std_train=np.std(arr_training_RVFL)
    roc_mean=np.mean(roc_matrix)
    roc_std=np.std(roc_matrix)
    PMSE_mean=np.mean(PMSE_matrix)
    PMSE_std=np.std(PMSE_matrix)
    print("AG-NBC测试集平均及标准差为", arr_mean_RVFL, arr_std_RVFL)
    print("AG-NBC训练集平均及标准差为",arr_mean_train,arr_std_train)
    print("AUC平均标准差为",roc_mean,roc_std)
    print("PMSE平均标准差为",PMSE_mean,PMSE_std)




