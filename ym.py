import numpy as np
import matplotlib.pyplot as plt 
# 定义数据集
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
# 定义标签（每个样本所属的分类）。
y = np.array([0, 0, 0, 1])
# 定义权重。对于单层感知器，权重通常初始化为0或者很小的数值。
# w = np.zeros(3)
w = np.random.random(3)
# 定义学习率。
eta = 0.1

for epoch in range(6):
    for x, target in zip(X, y):
        # 计算净输入 
        z = np.dot(w, x)
        # 根据净输入，计算分类值。
        y_hat = 1 if z >= 0 else 0
        # 根据预测值与真实值，进行权重调整。
        w = w + eta * (target - y_hat) * x
        # 注意数组的矢量化计算，相当于执行了以下的操作。
    #     w[0] = w[0] + eta * (y - y_hat) * x[0]
    #     w[1] = w[1] + eta * (y - y_hat) * x[1]
    #     w[2] = w[2] + eta * (y - y_hat) * x[2]
        print(target, y_hat)
        print(w)
        
