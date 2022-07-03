import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from collections import Counter, defaultdict
from minepy import MINE
import numpy as np
import pandas as pd
import operator
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
dx1=[0,0,0,0,1,1,1,2,2,2]
dx2=[1,2,2,2,1,1,3,4,4,4]
dy=[1,2,1,2,1,2,2,1,1,1]
nums_aac = defaultdict(int)
nums_aa = defaultdict(int)
nums_c = defaultdict(int)
lenght = len(dx1)
for i, j, z in zip(dx1, dx2, dy):
    nums_aac[(i, j, z)] += 1
    nums_aa[(i, j)] += 1
    nums_c[(z)] += 1
I_number = 0
for i in dx1:
    for j in dx2:
        for z in dy:
            if nums_aa[(i, j)] == 0 or nums_aac[(i,j,z)] == 0:
                I_number += 0
            else:
                I_number += (nums_aac[(i, j, z)] / lenght) * math.log(
                    (nums_aac[(i, j, z)] / lenght) / ((nums_aa[(i, j)] / lenght) * (nums_c[(z)] / lenght)))
print(I_number)