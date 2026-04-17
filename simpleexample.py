import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(2000)
y = np.random.rand(2000)
z = x + y  # just a smooth gradient

plt.scatter(x, y, c=z, cmap='inferno', s=9, alpha = 0.5, edgecolors='k', linewidths=0.3)
plt.colorbar(label='z')
plt.title('simple attempt')
plt.show()


