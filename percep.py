import random
import numpy as np
import matplotlib.pyplot as plt

file_path = 'ex4x.dat'
isAdmit_path = 'ex4y.dat'


def readFile(data_path, isAdmit_path):
    with open(data_path, 'r') as file:
        lines = file.readlines()

    data1 = []
    data2 = []

    for line in lines:
        column1, column2 = map(float, line.strip().split())
        data1.append(column1)
        data2.append(column2)

    with open(isAdmit_path, 'r') as file:
        lines = file.readlines()

    data = [float(line.strip()) for line in lines]

    numpy_data = np.array(data, dtype=np.float32)
    return data1, data2, numpy_data


# 读取数据
list1, list2, _train = readFile(file_path, isAdmit_path)

# 使用 zip 将两个列表的元素按行组合 结果为list形式
combined_list = list(zip(list1, list2))
# 将 list转换为 NumPy数组
X_train = np.array(combined_list)

# 特征归一化
mx = [0, 0]
mi = [1000, 1000]
for it in X_train:
    for i in range(2):
        mx[i] = max(mx[i], it[i])
        mi[i] = min(mi[i], it[i])

for i in range(80):
    X_train[i][0] = (X_train[i][0] - mi[0]) / (mx[0] - mi[0])
    X_train[i][1] = (X_train[i][1] - mi[1]) / (mx[1] - mi[1])

for i in range(len(_train)):
    if _train[i] == 0:
        _train[i] = -1

y_train = np.array(_train).reshape(len(_train), 1)

# 初始化参数
w = np.ones((1, 2))
b = 0

learning_rate = 0.005
epochs = 5000

loss_x, loss_y = [], []

# 训练感知机
for epoch in range(epochs):
    wrong = []
    loss = 0
    for i in range(len(X_train)):
        x = np.array(X_train[i]).reshape(2, 1)
        y_pred = np.dot(w, x) + b
        loss += (y_pred * y_train[i])
        if y_train[i] * y_pred < 0:
            wrong.append(i)  # 找出所有错误样本
    # 随机选择一个错误样本
    index = random.randint(0, len(wrong) - 1)
    xx = np.array(X_train[wrong[index]])
    yy = y_train[wrong[index]]

    loss_x.append(epoch + 1)
    loss_y.append(loss.item())

    # 更新梯度
    w += learning_rate * yy * xx
    b += learning_rate * yy

    # 打印损失值
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 为画图做准备 分离不同的点（在之后以颜色区分）
admitted_points_x, admitted_points_y, rejected_points_x, rejected_points_y = [], [], [], []
for i in range(len(y_train)):
    if (y_train[i] == 1):
        admitted_points_x.append(X_train[i][0])
        admitted_points_y.append(X_train[i][1])
    else:
        rejected_points_x.append(X_train[i][0])
        rejected_points_y.append(X_train[i][1])
plt.figure()
plt.scatter(admitted_points_x, admitted_points_y, color='green', label='Admitted')
plt.scatter(rejected_points_x, rejected_points_y, color='red', label='Not Admitted')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')

# 拟合曲线
x_line = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100)
y_line = (-w[0][0] * x_line - b) / w[0][1]
plt.plot(x_line, y_line, color='black', linestyle='dashed', label='Decision Boundary')
plt.legend()
plt.show()

# 画loss值更新图
plt.figure()
plt.plot(loss_x, loss_y)
plt.scatter(loss_x, loss_y, color='red')
plt.xlabel('iterations')
plt.ylabel('value of loss')
plt.show()
print(w)
print(b)
