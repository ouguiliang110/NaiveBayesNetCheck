import sklearn.preprocessing as pre_processing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

X = np.loadtxt('统计的数据集/abalone(1).txt')
print(X.shape[0], X.shape[1] - 1)
print(X[:,-1])
a=X[:,-1]
C1=[]
C2=[]
temp=0
temp1=0
temp2=0
for i in a:
    if i==6:
        C1.append(temp)
        temp1+=1
    elif i==7:
        C2.append(temp)
        temp2+=1
    temp+=1
C1=C1+C2
print(X[C1,:])
X=X[C1,:]
print(temp1,temp2)
print(X.shape[0], X.shape[1] - 1)

np.savetxt('统计的数据集/abalone(6,7).txt', X)
#print(X[:,-1][X[:,-1]==1])

'''
scaler = MinMaxScaler( )
scaler.fit(X)
my_matrix_normorlize=scaler.transform(X)
print(my_matrix_normorlize)
X=my_matrix_normorlize
X=X[0:500,:]



array=[3,7,12,15,18]
array_cut=[3,4,5,3,4]

tag=0
for i in array:
    X[:, i] = pd.cut(X[:, i], array_cut[tag], labels = range(array_cut[tag]))
    tag+=1


'''




'''
np.savetxt('统计的数据集/ring(dis).txt', X)

# 删除离散的列
p1 = np.array(array)
P1 = np.delete(X, p1, axis = 1)
np.savetxt('统计的数据集/ring(delete_dis).txt', P1)

#删除连续的列，相当于取离散的列
#需要加最后一列
P2=X[:,np.append(p1,X.shape[1] - 1)]
np.savetxt('统计的数据集/ring(delete_con).txt', P2)
'''












'''
X = pd.DataFrame(X)
print(X)
dum1 = pd.get_dummies(X[3], prefix = '4')
dum2 = pd.get_dummies(X[7], prefix = '3')
dum3 = pd.get_dummies(X[12], prefix = '6')
dum4 = pd.get_dummies(X[15], prefix = '9')
dum5 = pd.get_dummies(X[18], prefix = '18')

dum6 = pd.get_dummies(X[28], prefix = '24')
dum7 = pd.get_dummies(X[34], prefix = '28')
dum8 = pd.get_dummies(X[40], prefix = '38')


temp1=dum1.join(X.iloc[:, 1:4]).join(dum2).join(X.iloc[:, 5:8]).join(dum3).join(
    X.iloc[:, 9:12])


temp1 = X.iloc[:, 0:3].join(dum1).join(X.iloc[:, 4:7]).join(dum2).join(X.iloc[:, 8:12]).join(dum3).join(
    X.iloc[:, 13:15]).join(dum4).join(X.iloc[:,16:18]).join(dum5).join(X.iloc[:,19:21])

temp1 = X.iloc[:, 0:2].join(dum1).join(X.iloc[:, 3:5]).join(dum2).join(X.iloc[:, 6:8]).join(dum3).join(
    X.iloc[:, 9:11]).join(dum4).join(X.iloc[:, 17:20]).join(dum5).join(X.iloc[:, 21:28]).join(dum6).join(
    X.iloc[:, 29:34]).join(dum7).join(X.iloc[:, 35:40]).join(dum8).join(X.iloc[:, 41:45])


X = np.array(temp1)
print(temp1)
np.savetxt('统计的数据集/ring(one_hot).txt', X)

'''





























'''
print(X)
np.savetxt('统计的数据集/band.txt',X)
# 贝叶斯算法离散化后
array1 = np.zeros(shape = (0, X.shape[0]))
for n in range(0, m):
    k = 5
    d1 = pd.cut(vector_data[:, n], k, labels = range(k))
    array1 = np.vstack((array1, d1))
array1 = np.vstack((array1, label_data))
X1 = array1.T

'''
