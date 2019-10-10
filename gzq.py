%matplotlib inline
%pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation#绘制动态图

class Perceptron(object):#定义感知机函数
   
    def __init__(self, eta,epoch):#只要初始化，不要返回值
        self.eta = eta
        self.epoch = epoch  # 迭代次数

    def fit(self, X, y):
       
        self.w_ = np.zeros(1 + X.shape[1])  # weights, 初始值0,多加一个w0
        self.loss_ = []#存放错分次数

        # 循环更新
        for _ in range(self.epoch):#range(start, stop[, step])从0开始，到self.epoch次结束，步长为1
            loss = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))  #通过比较预测和真实数据矫正w
                self.w_[1:] += update * xi
                self.w_[0] += update
                loss += int(update != 0.0)
            self.loss_.append(loss)  # list.append(obj)，在list数组结尾添加对象
        return self

    def net_input(self, X):#根据输入数据代入公式
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):#与阀值比较进行预测
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
from matplotlib.colors import ListedColormap #用色块绘制分类边界
def plot_decision_regions(X, y, classifier, resolution=0.02):

        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))]) #np.unique 去除数组中的重复数字，排序之后输出

        # plot the decision surface 确定横纵轴边界
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # #X[:,k] 读取所有行第K列，最小-1, 最大+1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        #arange(start,end,step) 返回一个一维数组
        #meshgrid(x,y)产生一个以x为行，y为列的矩阵
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),#np.meshgrid()——生成网格点坐标矩阵
                         np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #调用predict()方法预测了grid矩阵当中的每一个点
        Z = Z.reshape(xx1.shape)#原数组重新组织

        # maps the different decision regions to different colors
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)#plt.contour画等高线
        plt.xlim(xx1.min(), xx1.max())#设定坐标上下限
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):#numerate()将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
             plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)




# 感知器类进行测试。
df = pd.read_csv('/home/zjr/下载/iris.data', header=None)
df.tail() 
y = df.iloc[0:100, 4].values#读取前100行的序号为4（第5列数据）
y = np.where(y == 'Iris-setosa', -1, 1)#标签分类：如果是 Iris-setosa y=-1否则就是1 
#.iloc[0:100,[0:2]] 读取前100行的前两列的数据，即两特征值
X = df.iloc[0:100, [0, 2]].values
p = Perceptron(eta=0.1, epoch=10)
p.fit(X, y)
p.loss_
print(p.loss_)
plt.figure(1)
#第一行第一列图形
ax1 = plt.subplot(2,2,1)
#第一行第二列图形
ax2 = plt.subplot(2,2,2)
#第二行
ax3 = plt.subplot(2,1,2)
#选择ax1
plt.sca(ax1)#观察数据是否线性可分
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')#scatter函数绘制散点图，两类用不同颜色和形状表示
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')#显示图例位置

plt.show
plt.sca(ax2)
plt.plot(range(1, len(p.loss_) + 1), p.loss_, marker='o')#绘制坐标图,横坐标为迭代次数，纵坐标为每次错分数
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.tight_layout()#tight_layout会自动调整子图参数，使之填充整个图像区域。它仅仅检查坐标轴标签、刻度标签以及标题的部分。
plt.sca(ax3)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()