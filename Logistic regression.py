import pandas as pd
import numpy as np
titanic= pd.read_csv('titanic.csv')

#print(titanic.shape)
#print(titanic)

##############################################################
# 由于该样本为简写，需要进行处理以展开频次（Freq）。下列语句展示了通过多次重复频次，衔接入原数组。
freq = titanic.Freq.to_numpy() # to_numpy()使得对象转化为数组，数组中有32个数。
index = np.repeat(np.arange(32),freq) # np.arange(32)生成了从0到32个数，np.repeat()将每个元素按照freq的频率进行重复。
#print(index.shape)
#print(index[:60])

titanic = titanic.iloc[index,:] # .iloc用于通过行号来取行数据，index代表行号，index实际上是[2,2,2...6,6,6...]，也即会抽取2行的全部列（，：）那么多次，再抽取6行全部列那么多次。
titanic = titanic.drop('Freq', axis=1)
#print(titanic.describe())

##############################################################
#直接利用python方法对数据进行分析
#print(pd.crosstab(titanic.Sex,titanic.Survived,normalize='index')) #normalize="index"表示按行进行标准化,也就是求出百分比。

##############################################################
#为开始回归进一步分类整理数据
from sklearn.model_selection import train_test_split # train_test_split即用于将数据分组为训练组和测试组的命令。
import statsmodels.api as sm
from patsy import dmatrices
# 在数据中存在分类变量时（如yes/no），dmatrices可以便捷的将其转化为数值型虚拟变量，并根据给出公式生成相应的数据矩阵X和响应变量向量Y。

train , test = train_test_split(titanic,test_size=0.3,stratify=titanic.Survived,random_state=0) # 以30%的比例划分测试组，stratify代表以‘Survived’变量为依据分组抽样。

y_train,x_train = dmatrices('Survived ~Class + Sex + Age', data =train,return_type = 'dataframe' )
# 从data=train的数据来源里，依据变量Survived转化的数值型虚拟变量，搭建响应变量Y。Class，Sex，Age生成X矩阵。
pd.options.display.max_columns = 10 #对象在显示时最多可以显示的列数，这里，它被设置为10。
#print(x_train.head()) # 展示后可以发现其去掉了多余的类别，比如（Male，Female）两类只保留了Male，这实现了降维去重。还增加了取值均为1的Intercept（截距项），我猜应该是为了避免全取0。
#print(y_train.head()) # 这里有两类，即Survived[No]和Survived[Yes]，只保留后者即可。
y_train=y_train.iloc[:,1] # 选择所有行（:表示所有行）的第二列（索引为1的列），并将这个选定的列（现在是一个Series）重新赋值给y_train。

y_test,x_test = dmatrices('Survived ~Class + Sex + Age', data =test,return_type = 'dataframe' )
y_test=y_test.iloc[:,1]

##############################################################
#采用statsmodels进行逻辑回归

model = sm.Logit(y_train,x_train) # statsmodels.api命令的Logit类创建实例model，以用于进行逻辑回归
results = model.fit() # 用fit方法进行估计拟合的结果（一条估计线
#results.params # 此时就估计出了β项，但对于非线性模型而言，想要考察边际应当转化β为对数几率（log odds），可参见pdf
print(np.exp(results.params)) #当乘客从成年（Age[T.Child]='0'）变为小孩时，其存活的新几率是原几率的3.09倍。
print(results.summary()) # a more aesthetically pleasing chart which could be applied into paper

##############################################################
# According to the results of the regression to predict
# First of all, we concern 训练误差
table = results.pred_table() # pred_table()方法生成混淆矩阵，感觉不如下面的pd.crosstab方法清晰
print(table) # table即混淆矩阵，实际未存活+预测未存活人数为949，实际存活+预测存活的人为245（？这个正类反类是不是标记错了）。第一行为实际为正类，第一列为预测为正类。
Accuracy = (table[0,0]+table[1,1]/np.sum(table)) #准确率(正确预测的百分比)-Accuracy rate；错分率 error rate = 1- AR
Sensitivity = table[1,1] / (table[1,0] + table [1,1]) # 灵敏度 ：即查准率，在实际为正的子样本中预测准确的比率。
Specificity = table[0,0] / (table[0,0] + table [0,1]) # 特异度 ：即（1-假阳率），在实际为负的子样本中预测准确的比率。

# Second, we concern 测试误差
prob = results.predict(x_test) #在测试样本中，用results估计函数预测每个个体的存活概率，记为prob
pred = (prob >= 0.5) #以0.5的概率为判断个体是否存活的门槛
table = pd.crosstab(y_test, pred, colnames=['predicted']) # 生成测试样本的混淆矩阵
table = np.array(table) # 为便于计算，转化为数组
# 可再行计算准确率，特异度，灵敏度等，并比较测试误差和训练误差。


print('******************我是分割线*****************我是分割线***********************')

##############################################################
#用sklearn模块进行逻辑回归，这样可以更好的画ROC图，计算AUC。

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,RocCurveDisplay
from sklearn.metrics import cohen_kappa_score

#sklearn进行逻辑回归
model = LogisticRegression(C=1e10) #sklearn模块只能估计带有惩罚项的逻辑回归，令惩罚非常接近0时即无差异于通常逻辑回归。此时生成的是逻辑回归的实例。
model.fit(x_train,y_train) # 用实例化的model去fit训练数据
print(model.coef_) # 其系数与statsmodels生成的系数差不多，只有截距项存在差异，但不重要

#sklearn计算准确率
print(model.score(x_test,y_test)) # 可以用score计算整体准确率，相当于跳过生成混淆矩阵

#sklearn对测试集进行预测
prob = model.predict_proba(x_test) # prob的数据是每个个体死亡与存活的概率，展示的是概率
pred = model.predict(x_test) # pred存储了预测后个体的结果，即每个个体存活与否的1或0

table = pd.crosstab(y_test, pred, rownames=['Actual'],colnames=['Predicted']) # 用pd展示混淆矩阵
print(table)

y_scores = prob[:, 1] #保留测试集正类的概率以用于下面计算
fpr, tpr, thresholds = roc_curve(y_test, y_scores) # 计算 ROC曲线的FPR和TPR
roc_display = RocCurveDisplay(fpr=fpr,tpr=tpr,estimator_name='ROC Curve') # 使用 RocCurveDisplay 绘制 ROC 曲线
roc_display.plot()
x=np.linspace(0,1,100) # 实例化一条线的始终点
plt.plot(x,x,'k--',linewidth=1) # 把这条线画进plt里
plt.title('ROC Curve (Test set)')
plt.show()

cohen_kappa_score(y_test,pred) #计算科恩的kappa指数，查看预测值与实际值之间的一致性。

