import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

##############################################################
# 首先生成模拟数据尝试进行支持向量机分类，这样填可以在低维空间直观展示SVM模型
x, y = make_blobs(n_samples=40, centers=2, n_features=2, random_state=6)
# 生成2个不同中心位置的正态分布，响应变量y取值为{0,1}（因为centers=2），根据支持向量机惯例将响应变量取值变为{-1,1}
y = 2*y - 1
data = pd.DataFrame(x,columns=['x1','x2']) # 为方便画图，将数据矩阵设为数据框
print(data)
sns.scatterplot(x='x1', y='x2', data=data, hue=y, palette=['blue','black'])
plt.show()
# hue=y表示根据响应变量y上色， palette=['blue','black']表示调色盘为蓝黑二色
model = LinearSVC(C=1000, loss='hinge', random_state=123) # 使用线性支持向量分类器进行分类
model.fit(x,y)
# C = 1000表示惩罚力度极大，几乎不允许差错；loss='hinge'表示使用合页损失函数，即标准的支持向量机，默认为loss='squared_hinge'，即合页损失的平方
dist = model.decision_function(x) # 计算观测值到分离超平面的“符号距离”
index = np.where(y * dist <= (1+1e-10)) # 找到位置索引，条件为比1略大一点的1e-10以防止计算偏差；
# 此即为条件 y*f(x) 以判断是否满足观测值为支持向量的条件。
print(index) # 支持向量的编号
print(x[index]) # 展示出支持向量


##############################################################
# 反复输入比较麻烦，定义两个函数。一是计算支持向量函数support_vectors()，二是绘制二维空间的支持向量函数svm_plot()
def support_vectors(model, x, y):
    dist = model.decision_function(x)
    index = np.where(y * dist <= (1+1e-10))
    return x[index]
def svm_plot(model,x,y):
    data = pd.DataFrame(x, columns=['x1','x2'])
    data['y'] = y
    sns.scatterplot(x='x1', y='x2', data=data,s=30, hue=y, palette=['blue','black'])
    s_vectors = support_vectors(model,x,y)
    plt.scatter(s_vectors[:,0], s_vectors[:,1], s=100, linewidths=1, facecolors='none', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy =np.meshgrid(np.linspace(xlim[0],xlim[1],50),
                        np.linspace(ylim[0],ylim[1],50))
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, colors='k', levels= [-1,0,1], alpha = 0.5,
                linestyles = ['--','-','--'])
    c=model.get_params()['C']
    plt.title(f'SVM(C={c})')
    plt.show()

svm_plot(model,x,y) # 由图可知，C=1000时惩罚力度很大，错误为0
model = LinearSVC(C=0.1, loss='hinge', random_state=123, max_iter=10000) # 再尝试惩罚参数 C= 0.1的SVM估计，最大迭代次数增加到10000次
model.fit(x,y)
print(support_vectors(model,x,y))
svm_plot(model,x,y) # 拓宽了间隔，但也增加了错误分类


##############################################################
#通过K折分类法交叉验证选定最优的惩罚参数C
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 1000]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
model = GridSearchCV(LinearSVC(loss='hinge', random_state=123, max_iter=10000),param_grid, cv=kfold)
model.fit(x,y)

print(model.best_params_) # 可得最优参数C=0.01
model = model.best_estimator_
print(len(support_vectors(model,x,y))) # 计算支持向量的数目为21


##############################################################
#接下来考察SVM类在非线性决策边界中的应用，
np.random.seed(1)
x = np.random.randn(200,2) # 生成来自标准正态分布的200个数据，有两维变量
y = np.logical_xor(x[:,0]>0 , x[:,1]>0) # 两个条件同对同错取值为False，
y = np.where(y, 1, -1) # 取值为True赋值为1，反之-1
data = pd.DataFrame(x, columns=['x1','x2'])

model = SVC(kernel='rbf', C=1, gamma=0.5, random_state=123)
model.fit(x,y)
print(model.n_support_) # 可得支持向量的数目，可知两类数据分别有46/47个
print(model.support_) # 可得支持向量的位置索引
model.support_vectors_ # 直接展示所有的支持向量
print(model.score(x, y))
plot_decision_regions(x, y, model, hide_spines = False)
plt.title('SVM(C=1,Gamma=0.5)')
plt.show()
# 考察惩罚参数C对决策边界的影响，可知C越大决策边界就会越受训练集的影响而扭曲，导致过拟合。反之则过分光滑。
# 考察gamma的影响，可知gamma越大，每个观测值的影响范围将变得很小，导致决策边界很容易完美区分，导致过拟合。
# 可以通过交叉验证的方式检验最优参数组合
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='rbf', random_state=123),param_grid, cv=kfold)
model.fit(x,y)
print(model.best_params_)



##################################################################
