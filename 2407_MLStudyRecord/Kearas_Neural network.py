import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist, reuters, boston_housing
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import keras

'''
(x_train,y_train),(x_test,y_test) = boston_housing.load_data(test_split=0.2, seed=113)
print(x_train.shape)

scaler = MinMaxScaler() # 归一化，使得最小值为 0，最大值为 1
scaler.fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

def set_my_seed(): # 先设定一系列随机数种子，以得到可重复的结果
    os.environ['PYTHONHASHSEED']='0' # 先设定os模块的Python..为0
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)

def build_model(): # 定义一个用于构建网络的函数
    model = Sequential() # 构建由一系列神经层线性排放而成的神经网络模型
    model.add(Input(shape=(x_train_s.shape[1],)))  # 使用 Input 层定义输入数据的形状，x_train_s.shape[1]=13，即13个特征变量
    model.add(Dense(units=256, activation='relu')) # 添加一个稠密层，即“全连接”层，含有256个神经元，使用ReLu激活函数
    model.add(Dense(units=256, activation='relu')) # 添加第二个隐藏层。
    model.add(Dense(units=1)) # 只有一个神经元的输出层
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    # 配置训练参数，使用RMSprop算法，以均方误差MSE为衡量效果的指标
    return model


model = build_model()
print(model.summary())

hist = model.fit(x_train_s, y_train,validation_split=0.25,
                 epochs=300, batch_size=16, shuffle=False)
# 选取25%的数据作为验证集， 训练300轮， batch_size=16即进行小批量梯度下降。输出结果为Keras的History对象。
Dict = hist.history
print(Dict.keys()) # 此字典包涵四个键，即验证集和训练集的损失函数与均方误差
val_mse = Dict['val_mse'] # 取出验证集均方误差
print(val_mse)
index = np.argmin(val_mse) # 找到最小的验证集均方误差的位置索引
print(index) # 第274个，即进行275轮可达到val_mse的最小值。
# 这是因为我们设定的神经网络较为复杂，而波士顿房价相对简单，容易出现过拟合，在此用“早停” 的方式防止过拟合。

# 更直观的，画图展示训练集与验证集的均方误差
plt.plot(Dict['mse'],'k',label='Train')
plt.plot(Dict['val_mse'],'b',label='Validation')
plt.axvline(index + 1, linestyle = '--', color = 'k')
plt.ylabel('MSE')
plt.xlabel('Mean Squared Error')
plt.legend()
plt.show()

# 利用最优轮数275（index+1）来重新训练神经网络模型
set_my_seed()
model = build_model()
model.fit(x_train_s, y_train,epochs= index+1,
                  batch_size=16, verbose =0) # 这里就不取出测试集了，并且不动态展示估计过程
print(model.evaluate(x_test_s,y_test)) # Keras的evaluate方法，类似于score()
# 两个值分别为测试集的损失函数loss和测试集的度量指标metrics，此例中二者都是均方误差（设定神经网络时定的）。
pred = model.predict(x_test_s) #预测结果pred为(102,1)矩阵
pred = np.squeeze(pred) # 变为(102,)的向量
np.corrcoef(y_test, pred) ** 2 # 测试集实际值y_test和pred的相关系数矩阵并平方之
print(np.corrcoef(y_test, pred)) # R2达到了0.883,拟合效果还不错


##############################################################
#使用Keras估计二分类问题，主要的差别在于设定神经网络模型时，输出层的激活函数应为逻辑函数，损失函数设为二值交叉熵
# def build_model(): # 定义一个用于构建网络的函数
#    model = Sequential() # 构建由一系列神经层线性排放而成的神经网络模型
#    model.add(Input(shape=(x_train_s.shape[1],)))
#    model.add(Dense(units=256, activation='relu'))
#    model.add(Dense(units=256, activation='relu'))
#    model.add(Dense(units=1, activation='sigmoid')) # 只有一个神经元的输出层
#    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # 损失函数设为二值交叉熵，以预测准确率accuracy为衡量效果的指标
#    return model

# *****神经网络正则化的方法，除了“早停”以外，还包括“丢包”和“惩罚（权重衰减）”，可参考p497，此处略
'''

#########################################################
#########################################################

'''
(x_trainval,y_trainval_original),(x_test,y_test_original) = reuters.load_data(num_words = 1000)
# x_trainval是一个含有8982个元素的一维数组。每个元素是一个列表。
print(x_trainval.shape)
print('x_trainval第一个元素是：')
print(x_trainval[0])

# 定义一个命令使列表转化为向量：
def vectorize_lists(lists, dimension=1000):
    results = np.zeros((len(lists),dimension))
    for i, list in enumerate(lists):
        results[i, list] = 1
    return results

x_trainval = vectorize_lists(x_trainval)
x_test = vectorize_lists(x_test)
y_trainval = to_categorical(y_trainval_original)
y_test = to_categorical(y_test_original)

print(y_trainval.shape, y_test.shape) # 二者均包含46列，代表不同新闻主题的46个虚拟变量
x_train, x_val, y_train, y_val = train_test_split(x_trainval,
                                                  y_trainval,stratify=y_trainval_original,test_size=1000,random_state=321)
def set_my_seed(): # 先设定一系列随机数种子，以得到可重复的结果
    os.environ['PYTHONHASHSEED']='0' # 先设定os模块的Python..为0
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)

set_my_seed()
def build_model(): # 定义一个用于构建网络的函数
    model = Sequential() # 构建由一系列神经层线性排放而成的神经网络模型
    model.add(Input(shape=(x_train.shape[1],)))  # 使用 Input 层定义输入数据的形状，x_train_s.shape[1]=13，即13个特征变量
    model.add(Dense(units=512, activation='relu')) # 添加一个稠密层，即“全连接”层，含有256个神经元，使用ReLu激活函数
    model.add(Dropout(0.25)) # 丢包层，随机丢掉一些神经元
    model.add(Dense(units=521, activation='relu')) # 添加第二个隐藏层。
    model.add(Dropout(0.25))
    model.add(Dense(units=46,activation='softmax')) # softmax为激活函数，所得结果为46个新闻主题的分类概率。
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # loss指定了损失函数，metrics指定了衡量算法指标。'categorical_crossentropy'是适合多分类问题的交叉熵损失函数。
    return model
model = build_model()
print(model.summary()) # param表示待估参数，该神经网络共计798766个可训练参数

hist = model.fit(x_train, y_train, validation_data=(x_val,y_val),
                 epochs=20, batch_size=64, shuffle=True)
print('fit结果为History对象，其键值如下：')
print(hist.history.keys()) # 与上例Dict.keys()用法一致，上文令hist.history = Dict.
# 其实就是四个关键键值:

####################################
# 通过验证集的损失函数(loss)检查最优参数
val_loss = hist.history['val_loss'] # 取出验证集的损失函数，val即 validation；
print(val_loss)
index_min = np.argmin(val_loss) # 找到最小的验证集【损失函数】的位置索引
print(index_min) # 第2个，即进行2轮可达到【损失函数】val_loss的最小值。

# 画图展示验证集的损失函数图像（Validation loss）
plt.plot(hist.history['loss'],'k',label='Training Loss')
plt.plot(val_loss,'b',label='Validation Loss')
plt.axvline(index_min, linestyle = '--', color = 'k')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
# 由图可知，虽然训练误差在不断下降，但验证误差在第二轮就达到最小，进一步训练反而提升验证误差。

####################################
# 通过验证集的预测准确率（accuracy）检查最优参数
val_accuracy = hist.history['val_accuracy']
index_max = np.argmax(val_accuracy)
print(index_max)
# 输出为7，即在第7轮时，验证预测集准确率可达最大值。画图展示之
plt.plot(hist.history['accuracy'],'k',label='Training Accuracy')
plt.plot(val_accuracy,'b',label='Validation Accuracy')
plt.axvline(index_max, linestyle = '--', color = 'k')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


prob = model.predict(x_test) # 预测每条新闻分别归属于46类主题的概率
print(prob.shape)
print(prob[0])
pred=model.predict(x_test)
classes_x=np.argmax(pred,axis=1)
# 预测每条新闻的主题归属
print(classes_x[:5]) # 展示前 5个预测结果

table = confusion_matrix(y_test_original, classes_x) #构建混淆矩阵
sns.heatmap(table, cmap='Blues') # 真懂哥都画图的
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
'''

#########################################################
#########################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('mnist数据形状为：（包括了60000个28*28的训练图像和10000个测试图像）')
print(x_train.shape,x_test.shape)

# 为了节省时间，只把原来的10000个测试集作为全样本，将10000条数据拆分为3000条训练集，3000条验证集和4000条测试集。
# 先把10000条原测试集拆成4000测试集和6000训练+验证
x_trainval, x_test,y_trainval, y_test = train_test_split(x_test, y_test,
                                                            stratify=y_test, test_size=0.4, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval,
                                                            stratify=y_trainval, test_size=0.5, random_state=369)
print('展示新训练集x_train的特征的最大值和最小值，其实就是像素的颜色')
print(np.min(x_train),np.max(x_train))
# 以下开始归一化，现将储存类型变成32位浮点数，再除以255
x_trainval = x_trainval.astype('float32')
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_trainval /= 255
x_train /= 255
x_val /= 255
x_test /= 255
# 四个变量至此变成了(3000*28*28)的格式的数据，即包含了3000个28*28像素矩阵三维数组。
# 然而Keras要求输入的是(28*28*1)的数组， 因此需要reshape()以增加维度
x_trainval = x_trainval.reshape((6000,28,28,1))
x_train = x_train.reshape((3000,28,28,1))
x_val = x_val.reshape((3000,28,28,1))
x_test = x_test.reshape((4000,28,28,1))

# 以上完成了对数据的处理，接着处理响应变量，使其成为虚拟变量的矩阵。
y_trainval = to_categorical(y_trainval)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

y_test_original = y_test
y_test = to_categorical(y_test)

def set_my_seed(): # 先设定一系列随机数种子，以得到可重复的结果
    os.environ['PYTHONHASHSEED']='0' # 先设定os模块的Python..为0
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)

set_my_seed()
def build_model(): # 定义一个用于构建网络的函数
    model =Sequential()
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy' ,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    # loss指定了损失函数，metrics指定了衡量算法指标。'categorical_crossentropy'是适合多分类问题的交叉熵损失函数。
    return model

model = build_model()
print(model.summary())

hist = model.fit(x_train, y_train, validation_data = (x_val,y_val), epochs=30,
                 batch_size=128, shuffle=False)
hist.history.keys()
val_loss = hist.history['val_loss']
index_min = np.argmin(val_loss)
print(index_min)
val_accuracy = hist.history['val_accuracy']
np.max(val_accuracy)
index_max = np.argmax(val_accuracy)
print(index_max)
# 得知验证集损失函数最小的索引是15，验证集准确率最高的索引是21

#接下来使用训练集+验证集的合集(trainval)重新估计该网络模型，并计算测试集的损失函数和预测准确率。
set_my_seed()
model = build_model()
model.fit(x_trainval, y_trainval, epochs=index_max + 1,
                 batch_size=128, shuffle=False)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose= 0)
print('以最佳轮数重新估计的模型，该模型在测试集上的表现（损失函数&预测准确率）：')
print(test_loss,test_accuracy)

prob = model.predict(x_test)
pred=model.predict(x_test)
classes_x=np.argmax(pred,axis=1)
print(classes_x[:5])

#懂哥再画个图，这次不用混淆矩阵命令
table = pd.crosstab(y_test_original, classes_x, rownames=['Actual'], colnames=['Predicted']) #构建混淆矩阵
sns.heatmap(table, cmap='Blues',annot=True, fmt='d') # 真懂哥都画图的
plt.tight_layout()
plt.show()