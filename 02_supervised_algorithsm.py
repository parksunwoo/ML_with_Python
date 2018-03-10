
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

plt.rcParams['figure.dpi'] = 300
# X, y = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(["클래스 0", "클래스 1"], loc=4)
# plt.xlabel("첫 번째 특성")
# plt.ylabel("두 번째 특성")
# print("X.shape: {}".format(X.shape))

X, y =mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("특성")
plt.ylabel("타깃")

