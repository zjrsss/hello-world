# 感知器的局限：如果两个类别的样本在空间中线性不可分，则感知器永远也不会收敛。
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 0])
# w = np.zeros(3)
w = np.random.random(3)
eta = 0.1

for epoch in range(70):
    for x, target in zip(X, y):
        z = np.dot(w, x)
        y_hat = 1 if z >= 0 else 0
        w = w + eta * (target - y_hat) * x
        print(target, y_hat)
        print(w)
        