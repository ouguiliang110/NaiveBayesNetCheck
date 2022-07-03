import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


X = np.loadtxt('[013]segment(0-1).txt')
X1=X[:,np.append(np.arange(9,18),[X.shape[1]-1])]




