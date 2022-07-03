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
dx=[0,0,0,0,1,1,1,2,2,2]
dy=[1,2,2,2,1,1,3,4,4,4]
nums_ac=defaultdict(int)
nums_a=defaultdict(int)
nums_c=defaultdict(int)
lenght=len(dx)
for i,j in zip(dx,dy):
    nums_ac[(i,j)]+=1
    nums_a[(i)]+=1
    nums_c[(j)]+=1
I_number=0
for i in dx:
    for j in dy:
        if nums_ac[(i,j)]==0:
            I_number+=0
        else:
            I_number += (nums_ac[(i, j)] / lenght) * math.log(
            (nums_ac[(i, j)] / lenght) / ((nums_a[(i)] / lenght) * (nums_c[(j)] / lenght)))
print(I_number)