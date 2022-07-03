import numpy as np
from minepy import MINE

x = np.linspace(0, 1, 1000)
y = np.sin(10 * np.pi * x) + x
mine = MINE(alpha = 0.6, c = 15)
mine.compute_score(x, y)

print("Without noise:")
print("MIC", mine.mic())

np.random.seed(0)
y += np.random.uniform(-1, 1, x.shape[0])  # add some noise
mine.compute_score(x, y)

print("With noise:")
print("MIC", mine.mic())