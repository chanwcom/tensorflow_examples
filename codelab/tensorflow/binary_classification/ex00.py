#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

example_size = 1000

class_label = np.random.randint(2, size=example_size)
class_0_idx = np.where(class_label == 0)
class_0_size = class_0_idx[0].shape[0]

class_1_idx = np.where(class_label == 1)
class_1_size = class_1_idx[0].shape[0]

x = np.zeros((example_size, 2))

mean0 = [0.0, 0.0]
cov0 = [[0.4, 0.0], [0.0, 1.0]]
x[class_0_idx[0], :] = np.random.multivariate_normal(mean0, cov0, class_0_size)


mean1 = [1.0, 2.0]
cov1 = [[1.0, 0.0], [0.0, 0.4]]
x[class_1_idx[0], :] = np.random.multivariate_normal(mean1, cov1, class_1_size)

plt.plot(x[class_0_idx[0], 0], x[class_0_idx[0], 1], 'r+')
plt.plot(x[class_1_idx[0], 0], x[class_1_idx[0], 1], 'k+')
plt.show()
