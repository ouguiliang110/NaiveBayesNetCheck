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

if __name__ == '__main__':
    X = np.loadtxt('../数据集/2satimage.txt')
    arr_test_RVFL = []
    arr_test_RVFL_1= []
    arr_test_RVFL_2=[]
    for k in range(0,10):
        train, test, num_class,train_index,val_index = prepare_data(X, 0.7)
        Rvf_one = RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
        Rvf_two = RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
        Rvf_three=RVFL(num_nides, regular_para, weight_random_range, bias_random_range, 'relu', False)
        # rcfl准确率
        B = train[1].astype(int)
        Rvf_one.train(train[0], B - 1, num_class)
        Rvfl_Pro_one = Rvf_one.predict(test[0], output_prob = True)
        result1 = np.argmax(Rvfl_Pro_one, axis = 1)
        acc1 = np.sum(np.equal(result1, test[1] - 1)) / len(test[1])
        arr_test_RVFL.append(acc1)
        print(acc1)

        data_train=X[:,:-1][train_index]
        label_train=X[:,-1][train_index]

        data_val = X[:,:-1][val_index]
        label_val = X[:,-1][val_index]
        Rvf_two.train(data_train,label_train.astype(int)-1,num_class)
        Rvfl_Pro_two = Rvf_two.predict(data_val, output_prob = True)
        result2 = np.argmax(Rvfl_Pro_two, axis = 1)
        acc2=np.sum(np.equal(result2,label_val-1))/len(label_val)
        arr_test_RVFL_1.append(acc2)

        data_train_1 = X[:, :-1][train_index]
        label_train_1 = X[:, -1][train_index]

        data_val_1 = X[:, :-1][val_index]
        label_val_1 = X[:, -1][val_index]
        Rvf_three.train(data_train_1, label_train_1.astype(int) - 1, num_class)
        Rvfl_Pro_three = Rvf_three.predict(data_val_1, output_prob = True)
        result3 = np.argmax(Rvfl_Pro_three, axis = 1)
        acc3 = np.sum(np.equal(result3, label_val_1 - 1)) / len(label_val_1)
        arr_test_RVFL_2.append(acc3)


    arr_mean_RVFL = np.mean(arr_test_RVFL)
    arr_std_RVFL = np.std(arr_test_RVFL)
    arr_mean_RVFL1=np.mean(arr_test_RVFL_1)
    arr_std_RVFL1=np.std(arr_test_RVFL_1)
    arr_mean_RVFL2 = np.mean(arr_test_RVFL_2)
    arr_std_RVFL2 = np.std(arr_test_RVFL_2)

    print("RVFL测试集平均及标准差为", arr_mean_RVFL, arr_std_RVFL)
    print("RVFL_one测试集平均值及标准差",arr_mean_RVFL1,arr_std_RVFL1)
    print("RVFL连续型测试集平均值及标准差",arr_mean_RVFL2,arr_std_RVFL2)





