import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

X = np.loadtxt('[018]musk01(0-1).txt')

#划分数组集

n=X.shape[0]
m=X.shape[1]-1


