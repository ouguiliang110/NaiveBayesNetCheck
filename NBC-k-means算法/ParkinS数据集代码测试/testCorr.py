import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
X=np.loadtxt('[021]parkinsons(0-1).txt')
X1=np.delete(X, X.shape[1] - 1, axis = 1)
df=pd.DataFrame(X1)
sns.pairplot(df)
plt.show()
print(df)