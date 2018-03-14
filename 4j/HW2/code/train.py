# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from BP import *

dataset = pd.read_table("spiral.txt",header=None, sep=' ')
X = dataset.iloc[:,[1,2]].values
Y = dataset.iloc[:,0].values

fig, axes = plt.subplots(1, 2)

nn = BP_network()  # build a BP network class
nn.CreateNN(2, 10, 1, 'Sigmoid')  # build the network

e = []
for i in range(1000):
    print(i)
    err, err_k = nn.TrainStandard(X, Y.reshape(len(Y), 1), lr=0.01)
    e.append(err)

axes[0].set_xlabel("epochs")
axes[0].set_ylabel("Error")
axes[0].set_title("Error change(lr=0.01)")
axes[0].plot(e)
# plt.show()

'''
draw decision boundary
'''
import numpy as np
import matplotlib.pyplot as plt

h = 0.01
x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
print(x0_min,x0_max,x1_min,x1_max)
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                     np.arange(x1_min, x1_max, h))


z = nn.PredLabel(np.c_[x0.ravel(), x1.ravel()])
z = z.reshape(x0.shape)

axes[1].contourf(x0, x1, z, cmap=plt.cm.Paired)
axes[1].scatter(X[:, 0], X[:, 1], c=Y)
axes[1].set_title("Two Spirals classification")

plt.show()








