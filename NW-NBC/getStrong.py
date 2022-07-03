from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

X = np.loadtxt("../数据集/[013]segment(0-1).txt")
m = X.shape[1] - 1  # 属性数量
print(m)
'''
X = X[:, m - 160:m+1]
m = X.shape[1] - 1
'''

n = X.shape[0]  # 样本数目 其中第一类1219，第二类683
print(n)
b=[9,10,11,12,16]
vector_data = X[:, b]
# 提取label类别
label_data = X[:, -1]
cases = vector_data
labels = vector_data
SetClass = set(X[:, m])
SetClass = list(map(int, SetClass))

K = len(SetClass)  # 类标记数量


ax1=plt.axes()
sns.heatmap(pd.DataFrame(vector_data).corr(),ax = ax1)
sns.pairplot(pd.DataFrame(vector_data))
ax1.set_title("neural-NBC")
plt.show()