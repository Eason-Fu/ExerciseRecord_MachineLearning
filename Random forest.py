import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay


##############################################################
#数据准备：由于波士顿房价数据在新版本中被取消了，所以直接在cmu官网下载数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# sep="\s+": 使用正则表达式 \s+ 作为分隔符，表示多个空格作为列的分隔符，因为这个数据集中的数据是由空格分隔的。
# skiprows=22: 跳过前22行，实际的数据从第23行开始，前面的数据是数据集的描述。
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
    ]
# 创建 DataFrame 并指定列名
x = pd.DataFrame(data, columns=column_names)
y = target


##############################################################
# 运用袋装法进行估计
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
model = BaggingRegressor(estimator
                         = DecisionTreeRegressor(random_state=123),
                         n_estimators=500, oob_score=True, random_state=0)
# 基学习器为 “DecisionTreeRegressor” 回归树；oob_score=True 保留袋外估计结果
model.fit(x_train,y_train)
pred_oob = model.oob_prediction_ # 得到袋外预测值
print(mean_squared_error(y_train, pred_oob)) # 袋外均方误差
print(model.oob_score_) # 袋外预测值拟合优度

print(model.score(x_test,y_test)) # 计算测试集的拟合优度
model2 = LinearRegression().fit(x_train,y_train)
print(model2.score(x_test,y_test)) # 如果用线性回归的方法，其拟合优度是小于袋装法的


##############################################################
#考察决策树数目(n_estimators)对袋外误差(MSE of OOB)的影响，运算比较费时故略去
#oob_errors = []
#for n_estimators in range(100,301):
#    model = BaggingRegressor(estimator
#                         = DecisionTreeRegressor(random_state=123),
#                         n_estimators=n_estimators, n_jobs= 16,
#                        oob_score=True, random_state=0)
#    model.fit(x_train,y_train)
#    pred_oob = model.oob_prediction_
#    oob_errors.append(mean_squaed_error(y_train,pred_oob))
#print(oob_errors)
#plt.plot(range(100,301),oob_errors)
#plt.xlabel('Number of trees')
#plt.ylabel('OOB errors')
#plt.title('Bagging OOB Errors')
#plt.show() #图片意味着，决策树的数目大于200后袋外误差基本稳定，继续扩大B不会增加袋外误差（即过拟合），但也不会下降误差


##############################################################
#随机森林估计
max_features = int(x_train.shape[1] / 3)
model = RandomForestRegressor(n_estimators=500,
                              max_features=max_features, random_state=0)
model.fit(x_train, y_train)
print(model.score(x_test,y_test)) # 随机森林的拟合优度略低于装袋法，因为随机森林其实是状态法的一种特例。
pred = model.predict(x_test) # 得到x_test里的预测值
plt.scatter(pred, y_test, alpha=0.6) # plt.scatter() 函数用于绘制散点图；alpha=0.6 设置了散点的透明度，使得重叠点更容易区分。
w = np.linspace(min(pred),max(pred),100) # 生成一个从 min(pred) 到 max(pred) 的等距序列，长度为 100。
plt.plot(w,w) # 绘制了一条直线，y = x，对应的点为 (w, w)；plot(a,b)即绘制（a,b）的点
plt.xlabel('pred')
plt.ylabel('y_test')
plt.title('Random Forest Prediction')
plt.show()


##############################################################
#预测变量重要性
print(model.feature_importances_)
#可以用画图的方式展示
sorted_index = model.feature_importances_.argsort()
plt.barh(range(x.shape[1]),   # plt.barh(y 轴上的位置,每个条形的长度);x.shape代表返回x的维度(500,15)，[1]代表取第二列，表示有15个特征；range()表示生成从0到x.shape[1]-1个数
         model.feature_importances_[sorted_index]) # 按照sorted_index（这是一个一维数组，0代表最重要，越大越不重要，如1,4,0,3,2....）分配给features得分，重新排列
plt.yticks(np.arange(x.shape[1]),x.columns[sorted_index])
plt.tight_layout
plt.show()


##############################################################
#由上可知，最重要的两个变量为LSTAT和RM，故绘制这两者的偏依赖图
features = [12,5]
PartialDependenceDisplay.from_estimator(model, x, features=features, n_jobs = -1)
plt.show()


##############################################################
#对于随机森林而言，决策树数目并不重要，关键变量是mtry，即每次选取的分裂变量max_features
scores = []
for max_features in range(1,x.shape[1]+1): # range生成从1到x特征数的数列1,2,3...13,14
     model = RandomForestRegressor(max_features=max_features,
                                   n_estimators=500, random_state=123)
     model.fit(x_train,y_train)
     score = model.score(x_test,y_test)
     scores.append(score)
index = np.argmax(scores) # argmax()即返回数组里最大值的索引，print(index)为8
print(range(1,x.shape[1]+1)[index]) # 取出index为8的数值，即0,1,2,3...8的数为9

plt.plot(range(1,x.shape[1]+1), scores,'o-')
plt.axvline(range(1,x.shape[1]+1)[index],linestyle='--',color='k',linewidth=1)
plt.show()

#上述针对测试集选择最优超参数mtry具有提前泄露测试集信息嫌疑，更严格的可以通过10折交叉验证选择最优参数
max_features = range(1,x.shape[1]+1)
param_grid = {'max_features':max_features} #将所有可能取值封装为字典形式
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(RandomForestRegressor(n_estimators=300,random_state=123),
                     param_grid, cv=kfold, scoring='neg_mean_squared_error',
                     return_train_score=True)
model.fit(x_train, y_train)
print(model.best_params_)
cv_mse = - model.cv_results_['mean_test_score']
plt.plot(max_features, cv_mse, 'o-')
plt.axvline(max_features[np.argmin(cv_mse)],linestyle='--',color='k',linewidth=1)
plt.show()