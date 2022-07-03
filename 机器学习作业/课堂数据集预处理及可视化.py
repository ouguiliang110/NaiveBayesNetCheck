import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
# 这里引入所用到的包
import warnings
from bubbly.bubbly import bubbleplot
from plotly.offline import init_notebook_mode, iplot



sns.set(style="white", color_codes=True)

# 读入数据
iris = pd.read_csv("ionosphere.csv") # the iris dataset is now a Pandas DataFrame
print(iris)
# 看下数据前5行
iris.head()

#这里设置x,y,z轴，气泡，气泡大小，气泡颜色分别代表6列~在二维平面想展示6个维度，除了x,y,z之外，
#只能用颜色，大小等代表其他维度了，bubbly还可以承受更高维度的数据，可以自己搜索
'''
sns.FacetGrid(iris, hue="y", size=5) .map(plt.scatter, "2", "3") .add_legend()
sns.jointplot(x="2", y="3", data=iris,kind="reg")
'''


f = iris.drop("1", axis=1).corr()
sns.heatmap(f, annot=True)
plt.show()
#展示图片