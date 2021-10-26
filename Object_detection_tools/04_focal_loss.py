import numpy as np


# alpha用于控制正负样本的权重
alpha = 0.25
y_true = np.array([1., 0., 1., 0.])
y_pred = np.array([0.95, 0.05, 0.5, 0.5])
alpha_weights = [alpha if y == 1 else 1-alpha for y in y_true]
print(alpha_weights)

pt = np.zeros(4)
index1 = np.argwhere(y_true == 1)
index0 = np.argwhere(y_true == 0)
# gam调制系数，用于控制难以分类样本的权重
gam = 2
pt[index1] = (1-y_pred[index1])**gam
pt[index0] = (y_pred[index0])**gam
weights = pt*alpha_weights
print(weights)