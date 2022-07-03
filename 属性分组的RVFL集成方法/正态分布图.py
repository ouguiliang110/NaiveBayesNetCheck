
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random




x = random.normal(loc=0.3093564432240637 , scale=0.05665474264050354,size=50000)
x1=random.normal(loc=0.34026168479112045, scale=0.06779537007126712,size=50000)
x2=random.normal(loc=0.35538187198481583, scale=0.06211532020881878,size=50000)
sns.kdeplot(x,bw = 0.05,shade = True,label = "Class1")
sns.kdeplot(x1,bw = 0.05,shade = True,label = "Class2")
sns.kdeplot(x2,bw = 0.05,shade = True,label = "Class3")
plt.show()



x6=random.normal(loc=0.29282853000486145 , scale=0.04519656661404224,size=50000)
x7=random.normal(loc=0.33211946980202467, scale=0.05804151797850572,size=50000)
x8=random.normal(loc=0.3850520001931139 , scale=0.05450723034297082,size=50000)

sns.kdeplot(x6,bw = 0.05,shade = True,label = "Class1")
sns.kdeplot(x7,bw = 0.05,shade = True,label = "Class2")
sns.kdeplot(x8,bw = 0.05,shade = True,label = "Class3")

plt.show()

x6=random.normal(loc=0.25282853000486145 , scale=0.047019656661404224,size=50000)
x7=random.normal(loc=0.34211946980202467, scale=0.05604151797850572,size=50000)
x8=random.normal(loc=0.410520001931139 , scale=0.0500723034297082,size=50000)

sns.kdeplot(x6,bw = 0.05,shade = True,label = "Class1")
sns.kdeplot(x7,bw = 0.05,shade = True,label = "Class2")
sns.kdeplot(x8,bw = 0.05,shade = True,label = "Class3")

plt.show()