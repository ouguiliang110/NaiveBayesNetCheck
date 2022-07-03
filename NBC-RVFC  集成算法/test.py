from itertools import combinations
import numpy as np
from sklearn.metrics import mutual_info_score
import pandas as pd
import  sklearn.preprocessing as pre_processing
from minepy import MINE
from sklearn import metrics



X = np.loadtxt('../数据集/[006]cmc(0-1).txt')

vector_data=X[:,:-1]
label_data=X[:,-1]

array1 = np.zeros(shape = (0, X.shape[0]))


'''
for n in range(0, m):
    k = 5
    d1 = pd.cut(vector_data[:, n], k, labels = range(k))
    array1 = np.vstack((array1, d1))
array1 = np.vstack((array1, label_data))
X1 = array1.T
'''

a=[0,1,1,1,0,1,2,1,1,1]

b=[0,2,0,1,1,0,1,2,2,1]



a=np.array(a)
a = a[:, np.newaxis]
X = np.hstack((vector_data[10:20,0][:,np.newaxis], a))


print(metrics.normalized_mutual_info_score(X[:,0], X[:,1]))

a=pd.DataFrame(X)
dum=pd.get_dummies(a[1],prefix = 'key')
new_df=a[[0.0]].join(dum)
new_np=np.array(new_df)

for i in range(0,4):
   result_NMI=metrics.normalized_mutual_info_score(new_np[:,0], new_np[:,i])
   print(result_NMI)

print('------------')
b=np.array(b)
b = b[:, np.newaxis]

print(vector_data[10:20,0])

Xb = np.hstack((vector_data[10:20,0][:,np.newaxis], b))

print(metrics.normalized_mutual_info_score(Xb[:,0], Xb[:,1]))

b=pd.DataFrame(Xb)
dum=pd.get_dummies(b[1],prefix = 'key')
new_df=b[[0.0]].join(dum)

new_np=np.array(new_df)
for i in range(0,4):
   result_NMI=metrics.normalized_mutual_info_score(new_np[:,0], new_np[:,i])
   print(result_NMI)


'''
vector_data[:,0]=pd.cut(vector_data[:,0],3,labels=range(3))
print(vector_data[:,0])
one_hot = pre_processing.OneHotEncoder()
X1=one_hot.fit_transform(vector_data[:,0]).toarray()
print(X1)

'''






