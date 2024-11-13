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
import cv2


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
    model.compile(loss='categorical_crossentropy' ,optimizer='adam', metrics=['accuracy'])
    # loss指定了损失函数，metrics指定了衡量算法指标。'categorical_crossentropy'是适合多分类问题的交叉熵损失函数。
    return model

model = build_model()
print(model.summary())

hist = model.fit(x_train, y_train, epochs=30,
                 batch_size=128, shuffle=False, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'测试集准确率: {test_acc}')

def prepare_image(image_path): # 实验发现这是最重要的影响预测结果的部分
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 加载灰度图
    img = cv2.resize(img, (28, 28))  # 调整图像大小
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)  # 二值化
    img = img.astype('float32') / 255  # 归一化
    img = img.reshape(1, 28, 28, 1)  # 调整形状
    return img

# 测试图片预测
image_files = ['test3.jpg', 'test7.png', 'test9.jpg']
for img_path in image_files:
    img = prepare_image(img_path)
    cv2.imshow('Input Image', img.reshape(28, 28))  # 展示图像
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    print(f"预测的类别是: {predicted_class}")