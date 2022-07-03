
import numpy as np
import sklearn.datasets as sk_dataset
import pandas as pd
import  sklearn.preprocessing as pre_processing
import random
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt



X = np.loadtxt('../数据集/[021]parkinsons(0-1).txt')
plt.scatter(X[:,0],X[:,1])
plt.show()
n=X.shape[0]
m=X.shape[1]

data=pd.DataFrame(X)
mean = data.mean()
std = data.std()
range_low = mean-3*std
range_high = mean+3*std
new_data = data
num=0
'''以3*detal准则为依据删除异常值'''
for i in range(n):  #行
    for j in range(m):  #属性
        if range_low[j] > data.iloc[i,j] or data.iloc[i,j] > range_high[j]:
            print('i',i)
            new_data = new_data.drop([i],axis=0)
            num = num+1
            print('num:',num)
            break
data = new_data
X=np.array(data)
print(X)
scaler = MinMaxScaler()
scaler.fit(X[:,:-1])
X[:,:-1]= scaler.transform(X[:,:-1])
print(X[:,-1])
plt.scatter(X[:,0],X[:,1])
plt.show()
'''数据分析查看'''
'''
data.describe()
data.plot(kind='box',subplots=True,layout=(4,6),sharex=False,sharey=False)
plt.show()
data.hist()
plt.show()
# 散点矩阵图
scatter_matrix(data)
pyplot.show()
'''
'''
data.to_excel('qinxidata.xlsx', sheet_name='Sheet0',index=False)
'''
