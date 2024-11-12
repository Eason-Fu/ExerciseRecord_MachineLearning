import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

##############################################################
#数据的准备工作。
iris = load_iris()
iris.feature_names = ['sepal_length','sepal_width','petal_length','petal_width'] # 原名有些繁琐，予以重新命名

x = pd.DataFrame(iris.data,columns=iris.feature_names) # 为考察特征变量之间的相关系数，将数据矩阵转化为数据框
#sns.heatmap(x.corr(),cmap='Blues',annot=True) # x.corr()用于查看数据框x的相关系数矩阵，heatmap画出了相关系数热区图以可视化
#plt.tight_layout() # 自动调整子图填充整个图像区域
#plt.show()

##############################################################
#应用sklearn模块对iris数据集进行全样本判别分析，并考察判别分析后的数据特征
y = iris.target
model = LinearDiscriminantAnalysis()
model.fit(x,y) # 用fit方法将实例化的线性判别model应用进数据集x，y上
model.score(x,y) # 看一下准确率

print(model.priors_) #考察各类别的先验概率，因为每一类别在样本中的出现次数相等，故各类别的先验概率均为0.333
print(model.means_) #考察4个特征变量在3个鸢尾花种的分组平均值
print(model.explained_variance_ratio_) #考察两个线性判元对与组间方差的贡献,ldd1解释了99.12%的组间方差

print('-----------------------------我是分割线---------------------------')
lda_loadings = pd.DataFrame(model.scalings_, index=iris.feature_names, columns=['LD1','LD2'])
print(lda_loadings)
# 上命令可以更好的观察两个线性判元的系数，model.scalings_可以考察线性判元的系数估计值。
# 线性判元就是由最优投影方向（w）组成的一组z=wx，因为此处原本数据分三组，降维后只有2个线性判元。
print('-----------------------------我是分割线---------------------------')
lda_scores = model.fit(x,y).transform(x) # 考察线性判别得分，有150行2列，每行对应每个观测值的线性判元ld1和ld2的得分。
print(lda_scores.shape)
print(lda_scores[:5,:]) #只看前五行的观测值
lda_scores = pd.DataFrame(lda_scores,columns=['ld1','ld2']) # 为画出线性判元散点图，先将得分转化为数据框
lda_scores['Species'] = iris.target # 加入响应变量Species
d = {0:'setosa',1:'versicolor',2:'virginica'} # 因为加入的变量Species是0,1,2，而非具体种类，所以映射的方法转为具体种类
lda_scores['Species'] = lda_scores['Species'].map(d) #用map方法转化
sns.scatterplot(x='ld1',y='ld2',data=lda_scores,hue='Species') #hue表示用颜色区分呢Species，如果用图形区分可以用style
plt.show()
# 从图可以看出，setosa可以很轻易的与另外两种区分开，而这种区分能力主要来自于ld1的角度。而从ld2的角度来看几乎无法区别三种花。

print('-----------------------------我是分割线---------------------------')
##############################################################
#进行线性判别分析(LDA)，为可视化分析选择绘制决策边界，仅选取2个特征变量。
x2 = x.iloc[:,2:4]
model = LinearDiscriminantAnalysis()
model.fit(x2,y) # 只保留petal_length，petal_width两个特征变量
model.score(x2,y) # 检查只用2个特征变量时预测的准确程度

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(np.array(x2),y,model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for LDA')
plt.show()

##############################################################
# 以上使用的是全样本，接下来采用sklearn随机划分为训练集和测试集。用训练集训练模型model，用predict方法
# 预测测试集的概率和结果，
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=123) # 以30%的比例划分测试组，stratify代表以‘Survived’变量为依据分组抽样。
model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)
model.score(x_test,y_test) # 建模用的是train，验证得分当然用test
prob = model.predict_proba(x_test)
pred = model.predict(x_test)
print(prob[:5])
print(pred[:5])
table = pd.crosstab(y_test, pred, rownames=['Actual'],colnames=['Predicted']) # 用pd展示混淆矩阵
print(table)
print(cohen_kappa_score(y_test,pred))

print('-----------------------------我是分割线---------------------------')
##############################################################
# 以下为使用sklearn进行二次判别分析（QDA），操作流程与上述线性判别分析类似。
model = QuadraticDiscriminantAnalysis()
model.fit(x_train,y_train)
model.score(x_test,y_test)
prob = model.predict_proba(x_test)
pred = model.predict(x_test)
confusion_matrix(y_test,pred)
cohen_kappa_score(y_test,pred)
# 经上述结果显示，对于iris数据集，QDA与LDA的预测效果相同。接下来仍使用2个特征变量，在全样本中绘制二次判别分析的决策边界。
model = QuadraticDiscriminantAnalysis()
model.fit(x2,y)
model.score(x2,y)
plot_decision_regions(np.array(x2),y,model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for QDA')
plt.show()
# 可知二次判别分析的决策边界为二次函数，即抛物线。

# 绘制决策边界时都是只选取了两个特征变量，难道不能多选择一些吗？