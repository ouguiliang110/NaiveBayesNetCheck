import numpy as np
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


def CountPNK(traingSet):
    vector_data = traingSet[:, :-1]
    label_data = traingSet[:, -1]
    dict_label = Counter(label_data)
    p_condition = defaultdict(float)
    K = len(dict_label)
    PNKArray=np.zeros((traingSet.shape[0],K))
    p=0
    for x in traingSet:
        print(x)
        PNKArray[p,np.int(x[-1]-1)]=1
        p+=1
    return PNKArray

def computePNK(trainingSet):
    nums_x=defaultdict(int)
    vector_data = trainingSet[:, :-1]
    label_data = trainingSet[:, -1]
    dict_label = Counter(label_data)
    K = len(dict_label)
    PNKArray = np.zeros((trainingSet.shape[0], K))

    for dx in trainingSet:
            dx= tuple(dx)
            nums_x[dx]+=1
    n=0
    for dx in trainingSet:
             dx=tuple(dx)
             PNKArray[n,int(dx[-1]-1)]=nums_x[dx] / dict_label[dx[-1]]
             n+=1
    return PNKArray

def Count(traingSet):
    smooth=1
    vector_data=traingSet[:,:-1]
    label_data=traingSet[:,-1]
    dict_label = Counter(label_data)
    p_condition=defaultdict(float)
    K=len(dict_label)
    ConditionArray = np.zeros((traingSet.shape[0], K,vector_data.shape[1]))
    #ConditionArray=[[]*K]*traingSet.shape[0]

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
            p_condition[(dx, key[0], key[1])] = (val + smooth / nums_sx) / (
                    nums_vd1[(key[1])] + smooth)
    for i in range(traingSet.shape[0]):
        for k in range(K):
            vector= vector_data[i,:]
            for j in range(vector_data.shape[1]):
                ConditionArray[i,k,j]=p_condition[(j,vector[j],k+1)]

    return ConditionArray



X = np.loadtxt('../?????????/[009]ecoli-3c.txt')

# my_matrix = np.loadtxt("../?????????/[010]glass(0-1).txt")
'''
# ?????????????????????????????????
scaler = MinMaxScaler()
scaler.fit(my_matrix)
my_matrix_normorlize = scaler.transform(my_matrix)
X = my_matrix_normorlize
print(my_matrix_normorlize)
''


'''
m = X.shape[1] - 1
print(m)
n = X.shape[0]  # ???????????? ???????????????1219????????????683
print(n)
# ?????????????????????
SetClass = set(X[:, m])
SetClass = list(map(int, SetClass))
print(SetClass)
K = len(SetClass)  # ???????????????

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
# ??????label??????
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
# ????????????????????????????????????4.1???????????????????????????????????????"S"??????0??????M"??????1???
# ??????????????????
vector_data=array
print(vector_data)
# ??????????????????????????????????????????????????????????????????????????????????????????1??????0????????????????????????
'''

NumClass = [0] * K
# ?????????U
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
    #Condition=Count(TrainSet)
    #PNKarray=CountPNK(TrainSet)
    num_x=computePNK(TrainSet)
    print(num_x)
    #print(Condition)

end = time.time()
print("?????????", str(end - start))
# ????????????
print(arr_test)
print(arr_train)
arr_mean = np.mean(arr_train)
arr_std = np.std(arr_train, ddof = 1)

arr_mean1 = np.mean(arr_test)
arr_std1 = np.std(arr_test, ddof = 1)

print("????????????????????????", arr_mean, arr_std)
print("????????????????????????", arr_mean1, arr_std1)