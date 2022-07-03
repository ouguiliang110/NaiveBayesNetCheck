import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from collections import Counter, defaultdict
from minepy import MINE
import numpy as np
import pandas as pd
import operator
import datetime
import time
from sklearn.preprocessing import MinMaxScaler

'''
X = np.loadtxt('[004]bd(0-1).txt')

m = X.shape[1]-1  # 属性数量
n = X.shape[0]  # 样本数目 其中第一类212，第二类55

SetClass=set(X[:,m])
SetClass=list(map(int,SetClass))
K = len(SetClass)  # 类标记数量

newarray=[np.zeros(shape=[0,m+1])]*20
for i in X:
    for j in SetClass:
        if i[m] == j:
            newarray[j] = np.vstack((newarray[j], i))

NewArray=np.zeros(shape=[0,m+1])
for i in SetClass:
    NewArray=np.vstack((NewArray,newarray[i]))




#统计各类数量
NumClass=[0]*K
# 初始化U
for i in X:
    for j in range(0,K):
        if i[m]==SetClass[j]:
            NumClass[j-1]=NumClass[j-1]+1

#数据集划分
train=[]
trainNum=0
val=[]
valNum=0
test=[]
testNum=0
for i in range(0,K):
    train.append(int(NumClass[i] * 0.5))
    trainNum+=int(NumClass[i] * 0.5)

    val.append(int(NumClass[i] * 0.2))
    valNum+=int(NumClass[i] * 0.2)

    test.append(NumClass[i] - train[i] - val[i])
    testNum+=NumClass[i] - train[i] - val[i]
train_index=[]
val_index=[]
test_index=[]
for i in range(0,K):
    idx=np.random.choice(np.arange(NumClass[i]), size = train[i], replace = False)
    train_index.append(np.array(idx))
    val_index.append(np.random.choice(np.delete(np.arange(NumClass[i]), train_index[i]), size =val[i], replace = False))
    test_index.append(np.delete(np.arange(NumClass[i]), np.append(train_index[i], val_index[i])))

'''


class NBayes(object):

    def __init__(self, smooth = 1):
        self.smooth = smooth  # 贝叶斯估计方法的平滑参数smooth=1，当smooth=0时即为最大似然估计
        self.p_prior = {}  # 先验概率
        self.p_condition = defaultdict(float)  # 条件概率
        self.AttributeIC = {}
        self.Wi=[]
        self.AttributeW = defaultdict(float)
        self.AttributeY = defaultdict(float)
        self.realCondition = defaultdict(float)

    def CountAttributeXY(self,dx,dy):
        nums_ac=defaultdict(int)
        nums_a=defaultdict(int)
        nums_c=defaultdict(int)
        lenght=len(dx)
        for i,j in zip(dx,dy):
                nums_ac[(i,j)]+=1
                nums_a[(i)]+=1
                nums_c[(j)]+=1
        I_number=0
        for i in dx:
           for j in dy:
               if nums_ac[(i, j)] == 0:
                   I_number += 0
               else:
                   I_number += (nums_ac[(i, j)] / lenght) * math.log(
                       (nums_ac[(i, j)] / lenght) / ((nums_a[(i)] / lenght) * (nums_c[(j)] / lenght)))
        return I_number
    def CountAttributeX1X2Y(self,dx1,dx2,dy):
        nums_aac=defaultdict(int)
        nums_aa=defaultdict(int)
        nums_c=defaultdict(int)
        lenght=len(dx1)
        for i, j, z in zip(dx1, dx2, dy):
            nums_aac[(i, j, z)] += 1
            nums_aa[(i, j)] += 1
            nums_c[(z)] += 1
        I_number = 0
        for i in dx1:
            for j in dx2:
                for z in dy:
                    if nums_aa[(i, j)] == 0 or nums_aac[(i, j, z)] == 0:
                        I_number += 0
                    else:
                        I_number += (nums_aac[(i, j, z)] / lenght) * math.log(
                            (nums_aac[(i, j, z)] / lenght) / ((nums_aa[(i, j)] / lenght) * (nums_c[(z)] / lenght)))
        return I_number


    def train(self, vector_data, label_data):
        n_samples = label_data.shape[0]  # 计算样本数
        # 统计不同类别的样本数，并存入字典，key为类别，value为样本数
        # Counter类的目的是用来跟踪值出现的次数。以字典的键值对形式存储，其中元素作为key，其计数作为value。
        dict_label = Counter(label_data)
        # print(dict_label[-1])
        mine = MINE(alpha = 0.6, c = 15)
        K = len(dict_label)
        for key, val in dict_label.items():
            # 计算先验概率
            self.p_prior[key] = (val + self.smooth / K) / (n_samples + self.smooth)
        # 计算后验概率
        # 分别对每个特征维度进行计算，vector_data.shape[1]为特征向量的维度
        for dx in range(vector_data.shape[1]):
            #F(ai,c)
            nums_vd = defaultdict(int)
            #F(c)
            nums_vd1=defaultdict(int)
            vector_dx = vector_data[:,dx]
            nums_sx=len(np.unique(vector_dx))
            for xd,y in zip(vector_dx,label_data):
                nums_vd[(xd,y)]+=1
                nums_vd1[(y)]+=1
            for key,val in nums_vd.items():
                self.p_condition[(dx,key[0],key[1])]=(val + self.smooth / nums_sx) / (
                                nums_vd1[(key[1])] + self.smooth)
        NI1=[]
        NI2=np.zeros((vector_data.shape[1],vector_data.shape[1]))
        all_relation=0
        all_relation1=0
        for i in range(vector_data.shape[1]):
                all_relation+=self.CountAttributeXY(vector_data[:,i],label_data)
        for i in range(vector_data.shape[1]):
            for j in range(vector_data.shape[1]):
               if  i!=j:
                all_relation1+=self.CountAttributeX1X2Y(vector_data[:,i],vector_data[:,j],label_data)

        for i in range(vector_data.shape[1]):
            NI1.append((self.CountAttributeXY(vector_data[:,i],label_data)*vector_data.shape[1])/all_relation)
        #NI(AJ,AK)

        '''
        for i in range(vector_data.shape[1]):
            for j in range(vector_data.shape[1]):
                if i!=j:
                    NI2[i,j]=(self.CountAttributeXY(vector_data[:,i],vector_data[:,j])*
                             vector_data.shape[1]*(vector_data.shape[1]-1))/all_relation1
        
        '''
        for i in range(vector_data.shape[1]):
            for j in range(vector_data.shape[1]):
                if i!=j:
                    NI2[i,j]=(self.CountAttributeX1X2Y(vector_data[:,i],vector_data[:,j],label_data)*vector_data.shape[1]*vector_data.shape[1]-1)/all_relation1
        #NI(Aj,Ak;C)

        Di=[0]*vector_data.shape[1]
        for i in range(vector_data.shape[1]):
              rel=0
              for j in range(vector_data.shape[1]):
                  if i!=j:
                    rel+=NI2[i,j]
              Di[i]=NI1[i]+(1/(vector_data.shape[1]-1))*rel
        for i in range(vector_data.shape[1]):
            self.Wi.append(1/(1+math.exp(-Di[i])))


    # 预测未知特征向量的类别
    def predict(self, input_v):
        p_predict = {}
        # y为类别，p_y为每个类别的先验概率
        for y, p_y in self.p_prior.items():
            p = p_y  # 计算每种后验概率
            for d, v in enumerate(input_v):  # 0 2 1 0 0 2 1 0
                # print(d, v)
                p *= self.p_condition[(d, v, y)]**self.Wi[d]
            p_predict[y] = p
        #     对字典按value进行排序
        p_predict_sorted = sorted(p_predict.items(), key = operator.itemgetter(1), reverse = True)
        #  print(p_predict.items())
        # 获取字典中value最大值所对应键的大小
        # return max(p_predict, key=p_predict.get)
        return p_predict_sorted[0]


if __name__ == "__main__":

    X = np.loadtxt('../数据集/[005]bp(0-1).txt')

   #my_matrix = np.loadtxt("../数据集/[010]glass(0-1).txt")
    '''
    # 将数据集进行归一化处理
    scaler = MinMaxScaler()
    scaler.fit(my_matrix)
    my_matrix_normorlize = scaler.transform(my_matrix)
    X = my_matrix_normorlize
    print(my_matrix_normorlize)
    ''


    '''
    m = X.shape[1] - 1
    print(m)
    n = X.shape[0]  # 样本数目 其中第一类1219，第二类683
    print(n)
    # 用两个类来完成
    SetClass = set(X[:, m])
    SetClass = list(map(int, SetClass))
    print(SetClass)
    K = len(SetClass)  # 类标记数量

    newarray = [np.zeros(shape = [0, m + 1])] * 20
    for i in X:
        for j in SetClass:
            if i[m] == j:
                newarray[j] = np.vstack((newarray[j], i))

    NewArray = np.zeros(shape = [0, m + 1])
    for i in SetClass:
        NewArray = np.vstack((NewArray, newarray[i]))
    print(NewArray)
    print(NewArray.shape)
    X = NewArray

    vector_data = X[:, :-1]
    # 提取label类别
    label_data = X[:, -1]

    # data = pd.DataFrame(vector_data)
    array1 = np.zeros(shape = (0, n))
    for n in range(0, m):
        k = 5
        d1 = pd.cut(vector_data[:, n], k, labels = range(k))
        array1 = np.vstack((array1, d1))
    array1 = np.vstack((array1, label_data))
    X1 = array1.T
    print(X1)
    array = X1

    '''
    array = np.zeros(shape = (0, m+1))
    p1=0
    for i in vector_data:
        k = 5
        d1 = pd.cut(i, k, labels = range(k))
        d1=np.append(d1,label_data[p1])
        p1+=1
        array = np.vstack((array, d1))
    # 以《统计学习方法》中的例4.1计算，为方便计算，将例子中"S"设为0，“M"设为1。
    # 提取特征向量
    vector_data=array
    print(vector_data)
    # 采用贝叶斯估计计算条件概率和先验概率，此时拉普拉斯平滑参数为1，为0时即为最大似然估
    '''

    NumClass = [0] * K
    # 初始化U
    p = 0
    for i in X:
        for j in range(0, K):
            if i[m] == SetClass[j]:
                NumClass[j] = NumClass[j] + 1
        p = p + 1
    print(NumClass)
    arr_train = []
    arr_test = []
    start = time.time()

    for k in range(0, 1):
        train = []
        trainNum = 0
        val = []
        valNum = 0
        test = []
        testNum = 0
        for i in range(0, K):
            train.append(int(NumClass[i] * 0.7))
            trainNum += int(NumClass[i] * 0.7)

            test.append(NumClass[i] - train[i])
            testNum += NumClass[i] - train[i]

        train_index = []
        val_index = []
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
            dividX.append(array[p2:p2 + NumClass[i], :])
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

        # print(valSet)
        bayes = NBayes(smooth = 1)
        bayes.train(TrainSet, Y)
        correct = 0
        for i in range(0, K):
            for j in trainSet[i]:
                if bayes.predict(j)[0] == SetClass[i]:
                    correct += 1
                # print(bayes.predict(j)[0], SetClass[i])
            # print(trainSet[i])
            # print(correct)

        arr_train.append(correct / trainNum)
        correct1 = 0
        for i in range(0, K):
            for j in testSet[i]:
                if bayes.predict(j)[0] == SetClass[i]:
                    correct1 += 1
                # print(bayes.predict(j)[0], SetClass[i])
            # print(testSet[i])
            # print(correct1)
        print(correct1 / trainNum)
        acc = correct1 / testNum
        arr_test.append(acc)
    end = time.time()
    print("时间是", str(end - start))
    # 求标准差
    print(arr_test)
    print(arr_train)
    arr_mean = np.mean(arr_train)
    arr_std = np.std(arr_train, ddof = 1)

    arr_mean1 = np.mean(arr_test)
    arr_std1 = np.std(arr_test, ddof = 1)

    print("训练集平均标准差", arr_mean, arr_std)
    print("测试集平均标准差", arr_mean1, arr_std1)
