import random
import numpy as np
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
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn import metrics
MatrixI=[]

def CountMutualXYC(vector_data,label_data):
    AttributeI = {}
    dict_label = Counter(label_data)
    MatrixI = np.zeros(shape = [vector_data.shape[1], vector_data.shape[1]])
    for dx in range(vector_data.shape[1]):
        for dy in range(vector_data.shape[1]):
            if dx != dy:
                W = defaultdict(int)  # 权重字典
                nums_vd = defaultdict(int)
                nums_vd1 = defaultdict(int)
                nums_vd2 = defaultdict(int)
                # 抽取特定维度
                vector_dx = vector_data[:, dx]
                vector_dy = vector_data[:, dy]
                # unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
                nums_sx = len(np.unique(vector_dx))  # 每个特征向量的可能取值个数
                nums_sy = len(np.unique(vector_dy))
                for xd, yd, y in zip(vector_dx, vector_dy, label_data):  # 取两个数，后取nums_vd用来标识数目
                    nums_vd[(xd, yd, y)] += 1  # 求F(ai,aj,c)
                    nums_vd1[(yd, y)] += 1
                    nums_vd2[(xd, y)] += 1
                    AttributeI[(dx, dy, y)] = 0.0
                for key, val in nums_vd.items():
                    AttributeI[(dx, dy, key[2])] += (nums_vd[(key[0], key[1], key[2])] / dict_label[
                        key[2]]) * (math.log((nums_vd[key[0], key[1], key[2]] / dict_label[key[2]]) / (
                            (nums_vd2[(key[0], key[2])] / dict_label[key[2]]) * (
                            nums_vd1[(key[1], key[2])] / dict_label[key[2]])),
                                             2))
    for dx in range(vector_data.shape[1]):
        for dy in range(vector_data.shape[1]):
            if dx!=dy:
                for i in range(len(dict_label)):
                    MatrixI[dx,dy]+=AttributeI[(dx,dy,i+1)]
    return MatrixI
def CountMutualXY(trainingSet):
    vector_data=trainingSet
    mine=MINE(alpha=0.8,c=15)
    MatrixI =np.zeros(shape = [vector_data.shape[1],vector_data.shape[1]])
    for dx in range(vector_data.shape[1]):
        for dy in range(vector_data.shape[1]):
            if dx!=dy:
                vector_dx=vector_data[:,dx]
                vector_dy=vector_data[:,dy]
                result_MI=metrics.normalized_mutual_info_score(vector_dx,vector_dy)
                MatrixI[dx,dy]=result_MI
    '''
    scaler = MinMaxScaler()
    scaler.fit(MatrixI)
    my_matrix_normorlize=scaler.transform(MatrixI)
    '''
    return MatrixI

IndependentMatrix=[]
RelevantMatrix=[]

def CountIndependenceMatrix(my_matrix):
    for x in range(my_matrix.shape[0]):
        for y in range(my_matrix.shape[1]):
            if x==y:
                my_matrix[x][y]=1
    print(my_matrix)
    add=my_matrix.sum(axis=1)
    min_index=np.argmin(add)
    IndependentMatrix.append(min_index)
    min_index_1=np.argmin(my_matrix[min_index])
    IndependentMatrix.append(min_index_1)
    print(IndependentMatrix)
    Begin_I=my_matrix[min_index,min_index_1]
    for i in range(my_matrix.shape[1]):
        Min_I=1
        Index_I=-1
        Add1=0
        for dy in range(my_matrix.shape[1]):
            if dy not in IndependentMatrix:
                IndependentMatrix.append(dy)
                combine = list(combinations(IndependentMatrix, 2))
                Add = 0
                for i in combine:
                    Add += my_matrix[i]
                #print(Add)
                #print(len(combine))
                Add = Add / len(combine)
                if (Add-Begin_I)<Min_I:
                    Min_I = (Add - Begin_I)
                    Add1=Add
                    Index_I=dy
                IndependentMatrix.pop()
        Begin_I =Add1
        print(Begin_I)
        print(Min_I)
        print(Index_I)
        if Index_I != -1:
           IndependentMatrix.append(Index_I)
    print(IndependentMatrix[0:int(0.7*my_matrix.shape[1])])

def CountRelevantMatrix(my_matrix):
    for x in range(my_matrix.shape[0]):
        for y in range(my_matrix.shape[1]):
            if x==y:
                my_matrix[x][y]=0
    print(my_matrix)
    add=my_matrix.sum(axis=1)
    max_index=np.argmax(add)
    RelevantMatrix.append(max_index)
    max_index_1=np.argmax(my_matrix[max_index])
    RelevantMatrix.append(max_index_1)
    print(RelevantMatrix)
    Begin_I=my_matrix[max_index,max_index_1]
    for i in range(my_matrix.shape[1]):
        Max_I=1
        Index_I=-1
        Add1=0
        for dy in range(my_matrix.shape[1]):
            if dy not in RelevantMatrix:
                RelevantMatrix.append(dy)
                combine = list(combinations(RelevantMatrix, 2))
                Add = 0
                for i in combine:
                    Add += my_matrix[i]
                #print(Add)
                #print(len(combine))
                Add = Add / len(combine)
                if (Begin_I-Add)<Max_I:
                    Max_I = Begin_I-Add
                    Add1=Add
                    Index_I=dy
                RelevantMatrix.pop()
        Begin_I =Add1
        print(Begin_I)
        print(Max_I)
        print(Index_I)
        if Index_I != -1:
           RelevantMatrix.append(Index_I)
    print(RelevantMatrix[0:int(0.7*my_matrix.shape[1])])


def CountGroup(my_matrix):
    add=my_matrix.sum(axis=1)
    print(add)
    min_index=np.argmin(add)
    print(min_index)
    IndependentMatrix.append(min_index)
    max_index=np.argmax(my_matrix[min_index])
    RelevantMatrix.append(max_index)

    for dx in range(my_matrix.shape[1]):
        if dx not in IndependentMatrix and dx not in RelevantMatrix:
                if compareRelevance(dx,my_matrix)==1:
                    IndependentMatrix.append(dx)
                elif compareRelevance(dx,my_matrix)==2:
                    RelevantMatrix.append(dx)

def compareRelevance(dx,my_matrix):
    Count1=0
    for y1 in range(len(IndependentMatrix)):
        Count1+=my_matrix[dx,IndependentMatrix[y1]]
    Count1/=len(IndependentMatrix)
    #print(Count1)
    Count2=0
    for y2 in range(len(RelevantMatrix)):
        Count2+=my_matrix[dx,RelevantMatrix[y2]]
    Count2 /= len(RelevantMatrix)
    #print(Count2)
    if Count1<Count2:
        return 1
    elif Count1>Count2:
        return 2


X = np.loadtxt('../数据集/[010]glass(0-1).txt')
# my_matrix = np.loadtxt("../数据集/[010]glass(0-1).txt")
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
    #print(TrainSet)
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
    print(TrainSet)
    '''
    getMatrixI=CountMutualXY(TrainSet)
    CountIndependenceMatrix(getMatrixI)
    CountRelevantMatrix(getMatrixI)
    '''
    MatrixI=CountMutualXYC(TrainSet,Y)
    print(MatrixI)
    CountIndependenceMatrix(MatrixI)
    CountRelevantMatrix(MatrixI)
