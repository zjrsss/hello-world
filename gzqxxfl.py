%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/zjr/下载/iris.data', header=None)
df.tail() 
y = df.iloc[0:100, 4].values#读取前100行的序号为4（第5列数据）
y = np.where(y == 'Iris-setosa', -1, 1)#标签分类：如果是 Iris-setosa y=-1否则就是1 
#.iloc[0:100,[0:2]] 读取前100行的前两列的数据，即两特征值
X = df.iloc[0:100, [0, 2]].values
import matplotlib.pyplot as plt
#画散点图显示数据
plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='red', marker='x', label='vesetosa')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.title('Original Data')
plt.show()
#归一化处理
# 均值
#axis 不设置值，对 m*n 个数求均值，返回一个实数
#axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
#axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵

u = np.mean(X, axis=0)
# 方差
v = np.std(X, axis=0)

X = (X - u) / v

# 作图
plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='o', label='Positive')
plt.scatter(X[50:, 0], X[50:, 1], color='red', marker='x', label='Negative')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.title('Normalization data')
plt.show()

#直线初始化
# X加上偏置项
X = np.hstack((np.ones((X.shape[0],1)), X))#X.shape[0]为X第一维的长度，np.ones建立全1矩阵；hstack水平将两个数组水平组合
# 权重初始化，符合标准正态
w = np.random.randn(3,1)
s = np.dot(X, w)
y_pred = np.ones_like(y)    # 预测输出初始化，返回一个用1填充的跟输入形状和类型一致的数组
loc_n = np.where(s < 0)[0]   # 小于零索引下标，然后更新预测值
y_pred[loc_n] = -1
# 第一个分类错误的点
t = np.where(y != y_pred)[0][0]
# 更新权重w
eta=0.1
w += eta * ( y[t]- y_pred[t])* X[t, :].reshape((3,1))

for i in range(100):
    s = np.dot(X, w)
    y_pred = np.ones_like(y)
    loc_n = np.where(s < 0)[0]
    y_pred[loc_n] = -1
    num_fault = len(np.where(y != y_pred)[0])
    print('第%2d次更新，分类错误的点个数：%2d' % (i, num_fault))
    if num_fault == 0:
        break
    else:
        t = np.where(y != y_pred)[0][0]
        w += eta * ( y[t]- y_pred[t])* X[t, :].reshape((3,1))
# 直线第一个坐标（x1，y1）
x1 = -2
y1 = -1 / w[2] * (w[0] * 1 + w[1] * x1)
print(y1)
# 直线第二个坐标（x2，y2）
x2 = 2
y2 = -1 / w[2] * (w[0] * 1 + w[1] * x2)
print(y2)
# 作图
plt.scatter(X[:50, 1], X[:50, 2], color='blue', marker='o', label='Positive')
plt.scatter(X[50:, 1], X[50:, 2], color='red', marker='x', label='Negative')
plt.plot([x1,x2], [y1,y2],'r')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()
