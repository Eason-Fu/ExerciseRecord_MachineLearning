import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


##############################################################
#数据的准备工作。
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['diagnosis']=cancer.target
d = {0:'malignant',1:'benign'}
df['diagnosis'] = df['diagnosis'].map(d)

print(df.shape)
pd.options.display.max_columns = 40
print(df.head(2))
print(df.iloc[:,:3].describe()) # 为节省空间，考察前3个特征变量的统计特征
print(df.diagnosis.value_counts(normalize=True)) # 查看良性和恶性肿瘤诊断的占比

x , y = load_breast_cancer(return_X_y=True) # 再次载入数据，但是利用load_breast_cancer命令直接返回数据矩阵x和响应变量y。
x_train , x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=100, random_state=1)

##############################################################
# 由于KNN方法要求变量的变化幅度相近，所以需要使用StandardScaler类对特征变量进行标准化，即使其变为均值为0而
# 标准差为1的正态分布。
scaler = StandardScaler()
scaler.fit(x_train) # 根据x_train计算变量均值和标准差
x_train_s = scaler.transform(x_train) # 应用transform()方法转化训练集和测试集
x_test_s = scaler.transform(x_test)
np.mean(x_train_s, axis=0)
np.std(x_train_s,axis=0) #考察一下转化后的结果
np.mean(x_test_s, axis=0)
np.std(x_test_s,axis=0) # 标准化后的测试集其实变量均值并不为0，标准差也不为1， 因为转化用的值fit自train集。
# 但这正是我么要的效果，否则相当于提前泄露了测试集的信息，可能导致偏差。
print('-----------------------------我是分割线---------------------------')

##############################################################
#构建K近邻法的模型，寻求能使预测率最高的K。
model = KNeighborsClassifier(n_neighbors=5) #先进行K=5的KNN估计，在30维的特征空间里找邻居
model.fit(x_train_s,y_train)
pred = model.predict(x_test_s)
pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['Predicted']) #计算混淆矩阵
model.score(x_test_s,y_test) #预测的准确率，此时已达到0.97
print('-----------------------------我是分割线---------------------------')
scores= []
ks =range(1,51)
for k in ks: #for循环遍历ks
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_s,y_train)
    score = model.score(x_test_s,y_test) # 此处其实是相当于通过测试集选择了最优的参数K，颇有泄露信息之嫌。
    scores.append(score) # 把得到的准确率写入scores列表

index_max = np.argmax(scores) # 返回scores存储的最大值的索引
print(f'Optimal K:{ks[index_max]}') #这样就展示出最佳K值
print('-----------------------------我是分割线---------------------------')
plt.plot(ks,scores,'o-') # 用绘图的方式更直观的展现
plt.xlabel('k')
plt.axvline(ks[index_max],linewidth=1,linestyle='--',color='k')
plt.ylabel('Accuracy')
plt.title('KNN')
plt.tight_layout
plt.show()

#因此下内容对训练集进行10折交叉验证，以选择最优超参数K
param_grid = {'n_neighbours':range(1,51)} # 以字典的形式定义超参数K的网格
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) #定义10折分层随机分组
model = GridSearchCV(KNeighborsClassifier(),param_grid,cv=kfold)
model.fit(x_train_s,y_train)
print(model.best_params_)