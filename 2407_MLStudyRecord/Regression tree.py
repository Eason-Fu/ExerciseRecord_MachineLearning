import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor, export_text

from sklearn.metrics import cohen_kappa_score

#Boston = load_boston()
#x_train , x_test, y_train, y_test = train_test_split(Boston.data, Boston.target, test_size=0.3, random_state=0)
#model = DecisionTreeRegressor(max_depth=2, random_state=123) # 限制回归树的最大深度为2，该命令使用的是改进的CART算法，节点分裂时可能存在多种分裂方法，故明确seed。
#model.fit(x_train, y_train) # 配置后有参数ccp_alpha = 0.0，意即成本复杂性参数为0，即不惩罚决策树的规模。
#model.score(x_test, y_test)

##############################################################
# 打印展示出文本格式的决策树
#print(export_text(model,feature_names=list(Boston.feature_names)))
#plot_tree(model, feature_names=Boston.feature_names, node_ids=True, rounded=True, precision=2)
#'feature_names'提供特征变量的名称，'node_ids'将节点编号，'rounded'节点四周为圆角，'precision'表示精确到小数点后两位。

##############################################################
# 分类树-数据处理（观察了解源数据，转化为数值型的分类矩阵）
bank = pd.read_csv('bank-additional.csv',sep=';')
print(bank.shape)
print(bank.head(5)) # 看一下原始数据
bank = bank.drop('duration',axis=1) # duration表示自上次去电后过了多少秒，无意义，可去除
print(bank.y.value_counts()) # 考察样本中有购买金融产品意愿的人数与比例。

print('-----------------------------我是分割线---------------------------')

x_raw = bank.iloc[:,:-1] # 由于原bank文件存在大量的'字符型分类变量'，需要转化成'数值型'虚拟变量才能进一步操作
x = pd.get_dummies(x_raw) # 根据特征向量矩阵x_raw生成虚拟变量矩阵x
print(x.head(2))
y = bank.iloc[:,-1] # 生成由虚拟变量组成的y
x_train , x_test, y_train, y_test = train_test_split(x, y, stratify=y,test_size=1000, random_state=1)

##############################################################
# 分类树-开始进行分类
model = DecisionTreeClassifier(max_depth=2, random_state=123)
model.fit(x_train,y_train)
print(model.score(x_test, y_test)) # 准确率为90.4%，但考虑到样本中已有89.1%的顾客不愿意购买金融产品，此准确率并不高
# 为了更直观的展示，接下来需画出决策树
plot_tree(model,feature_names=x.columns, node_ids=True, rounded=True, precision=2)
plt.show()

##############################################################
# 为选择最佳分类树的规模，考察成本复杂性参数ccp_alpha对叶节点总不纯度的影响，使用cost_complexity_pruning_path()方法
# impurities即不纯度，我们希望每次分裂后不纯度下降最多。
# ccp_alpha,即成本复杂性参数，0时代表没有惩罚项，树的规模最大，导致过拟合；1时只剩下树的主干。
# 我们并不希望决策树过于复杂，因此需要用ccp_alpha控制对决策树规模的惩罚力度，这可通过交叉验证实现，即“成本复杂性修枝”。
model = DecisionTreeClassifier(random_state=123)
path = model.cost_complexity_pruning_path(x_train, y_train)
#画图展示ccp_alpha与叶节点总不纯度的关系：
plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle='steps-post')
plt.xlabel('alpha(cost-complexity parameter)')
plt.ylabel('Total Leaf Impurities')
plt.title('Total Leaf Impurities vs alpha for Training Set')
max(path.ccp_alphas), max(path.impurities)
plt.show()
# 观察此图，alpha为0时不惩罚决策树，每个观测值就是一个叶节点，所以不纯度为0；
# alpha上升时，惩罚力度增加，alpha接近0.3时决策树仅剩下树桩，不纯度达到最大。接下来要选择最优的ccp_alpha

param_grid = {'ccp_alpha':path.ccp_alphas} # 以字典形式创建参数alpha网络
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold) #传入参数alpha网格和10折分层随机分组
model.fit(x_train, y_train)
print(model.best_params_) # 展示最优的ccp_alpha
model = model.best_estimator_ # 将实例model重新定义为相应的最优模型
print(model.score(x_test, y_test)) # 最优的其实还是90.4%
# 画图展示，'proportion=True'代表显示观测值的比重
plot_tree(model,feature_names=x.columns, node_ids=True,proportion=True, rounded=True, precision=2)
plt.show()
# 根节点 #node=0 拥有100%的数据，89%的样本无意购买，11%样本有意购买。划分标准为公司雇员数是否小于5087.65人。
# node#1 包含12.6%的样本，满足自上次电联天数小于12.5日的进入左侧node#2。
# node#2包括2.8%的样本，其中有购买意愿的顾客占73%。可以被认为是可以致电的潜在客户。

#展示变量的重要性并绘图观测
print(model.feature_importances_) #只有三个变量是重要的，因为分裂决策树时只用了三个。
sorted_index = model.feature_importances_.argsort()
plt.barh(range(x_train.shape[1]),
         model.feature_importances_[sorted_index])
plt.yticks(np.arange(x_train.shape[1]),
          x_train.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree')
plt.tight_layout()
plt.show()


##############################################################
# 模型已构筑完，接下来测度灵敏度等指标
pred = model.predict(x_test)
pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['Predicted'])
print(model.score(x_test,y_test))
print(cohen_kappa_score(y_test,pred))
# 预测准确率是90.4%，但无购买意愿的客户即占比89.1%。而灵敏度只22%，即只能成功识别22%有购买意愿的顾客。kappa指标为0.296,一致性也一般。

# 以上预测默认以可能性大于0.5作为预测的标准，为提高灵敏度以识别更多有意向购买的潜在客户，可以选择降低门槛值。
prob = model.predict_proba(x_test)
print(model.classes_) #确认两列的概率：第一列为无意向购买，第二列为有意向购买

