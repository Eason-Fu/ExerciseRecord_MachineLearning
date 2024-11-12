import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from PIL import Image



digits = load_digits() # 返回的是sklearn的Bunch数据
dir(digits) # 考察Bunch数据的各个部分，包括'images','data'等，前者即存储了1797个8*8的图片，后者将其扁平化为1797*64
print(pd.Series(digits.target).value_counts()) # 考察样本数字的分布，发现10个数字大致出现频次相同
plt.imshow(digits.images[88], cmap=plt.cm.gray_r) # 随意看一下第8张图片
plt.show()

x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2,
                                                     random_state = 0)

model = SVC(kernel="linear",random_state=123) # 默认的线性核SVM估计
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

model = SVC(kernel="poly", degree=2, random_state=123) # 二次多项式核SVM估计
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

model = SVC(kernel="poly", degree=3, random_state=123) # 三次多项式核SVM估计
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

model = SVC(kernel="rbf", random_state=123) # 默认的径向核SVM估计
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

model = SVC(kernel="sigmoid", random_state=123) # S型核SVM估计
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
# 结果显示，除了S型核SVM估计效果不佳以外，其他核的SVM估计效果均不错。
# 接着用径向核验证选择最优参数组合。

model = SVC(kernel="rbf",C=1, gamma=0.001, random_state=123)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

image_files = ['test3.png', 'test7.png', 'test9.jpg']

# 准备一个列表存储处理好的图像数据
all_images_data = []

for img_file in image_files:
    img = Image.open(img_file).convert('L')  # 将图片转为灰度
    img = img.resize((8, 8), Image.Resampling.LANCZOS)

    plt.imshow(img, cmap='gray') # 使用灰度色彩显示图像
    plt.show()

    # 将图像转化为模型输入格式（8x8），并将像素值缩放为与digits一致的范围
    img_data = np.array(img) # 将处理好的图像转化为一个 NumPy 数组。此时数组的大小为 8x8，每个元素表示一个像素的灰度值（0 表示黑，255 表示白
    img_data = 16 - img_data // 16
    # digits 数据集的像素值范围是 0 到 16，而 PIL 图像的像素范围是 0 到 255。所以这里我们将像素值从 0-255
    # 缩小到 0-16 范围。16 - img_data 是为了翻转颜色（因为在 digits 数据集中，黑色是 16，白色是 0.
    img_data = img_data.flatten()  # 展平为一维数组以符合模型输入格式

    all_images_data.append(img_data)

all_images_data = np.array(all_images_data)

predictions = model.predict(all_images_data)

for idx, img_file in enumerate(image_files):
    print(f"模型预测的 {img_file} 是数字: {predictions[idx]}")