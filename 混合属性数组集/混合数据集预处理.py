
import sklearn.preprocessing as pre_processing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def transform_one_hot(labels):
  n_labels = np.max(labels) + 1
  one_hot = np.eye(n_labels)[labels]
  return one_hot

X=np.loadtxt('../数据集/[008]band(0-1).txt',delimiter=",")
X1=pd.read_table('../数据集/[008]band(0-1).txt',delimiter = ',')
print(X1)

print(X1['1'])
label=pre_processing.LabelEncoder()


X1['19']=label.fit_transform(X1['19'])

X1['17']=label.fit_transform(X1['17'])

X1['20']=label.fit_transform(X1['20'])

print(X1['17'])
print(X1)

X2=np.array(X1)
print(X2)
print(X2[:,16])
print(X2[:,18])

'''
# 将数据集进行归一化处理
scaler = MinMaxScaler()
scaler.fit(X2[:,1:-1])
X2[:,1:-1] = scaler.transform(X2[:,1:-1])
'''

np.savetxt('adult(1).txt',X2[0:5000])



'''
X2=pd.DataFrame(X2)
X2[0]=pd.get_dummies(X2[0],prefix = 'key')
print(X2)
X2=np.array(X2)

'''







