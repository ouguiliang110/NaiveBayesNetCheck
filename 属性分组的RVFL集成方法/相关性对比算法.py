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

def max2(x):
    xp=x.copy()
    xp[xp == 1] = 0.002
    max=np.max(xp)
    return max

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
X = np.loadtxt('../数据集/[013]segment(0-1).txt')
Xp = pd.DataFrame(X[:,:-1])

Xp.rename(columns = {0: 'A1', 1: '', 2: 'A3', 3: '', 4: 'A5', 5: '', 6: 'A7', 7: '', 8: 'A9', 9: '',10: 'A11', 11: '', 12: 'A13', 13: '', 14: 'A15', 15: '', 16: 'A17', 17: '', 18: 'A19'},
              inplace = True)


Xp_corr=abs(Xp.corr())
Xp=np.array(Xp)
max=max2(Xp)
print(max)
min=np.min(Xp)
print(min)


print(Xp)
sns.heatmap(Xp_corr, annot = False, vmin = min, vmax = max, cmap = "hot_r",
                    annot_kws = {'size': 8, 'weight': 'bold'})
plt.show()


Xp1=pd.DataFrame(X[:,[5,6,7,8]])

Xp1.rename(columns = {0: 'A6', 1: 'A7', 2: 'A8', 3: 'A9'},
              inplace = True)


Xp1=abs(Xp1.corr())
Xp2=pd.DataFrame(X[:,[9,10,11,12,14,16]])

Xp2.rename(columns = {0: 'A10', 1: 'A11',2: 'A12', 3: 'A13', 4: 'A15',5:'A17'},
              inplace = True)


Xp2=abs(Xp2.corr())
sns.heatmap(Xp1, annot = True, vmin = min, vmax = max, cmap = "hot_r",
                    annot_kws = {'size': 8, 'weight': 'bold'})
plt.show()

sns.heatmap(Xp2,annot = True, vmin = min, vmax = max, cmap = "hot_r",
                    annot_kws = {'size': 8, 'weight': 'bold'})
plt.show()



Xp3=pd.DataFrame(X[:,[9,10,11,12,14,16,8]])


Xp3.rename(columns = {0: 'A10', 1: 'A11',2: 'A12', 3: 'A13', 4: 'A15',5:'A17',6:'A9'},
              inplace = True)


Xp3=abs(Xp3.corr())
sns.heatmap(Xp3, annot = True, vmin = min, vmax = max, cmap = "hot_r",
                    annot_kws = {'size': 8, 'weight': 'bold'})
plt.show()







