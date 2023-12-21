import numpy as np
import matplotlib.pyplot as plt
import torch

file_path = 'data/ex4x.dat'
isAdmit_path = 'data/ex4y.dat'


def readFile(data_path, isAdmit_path):
    # file.readlines()读取文件的所有行并将其存储为一个列表，其中每个元素都是文件中的一行
    with open(data_path, 'r') as file:
        lines = file.readlines()

    data1 = []
    data2 = []

    """
    strip() 方法用于去除字符串两端的空白字符，包括空格、制表符和换行符;
    在 line.strip() 的基础上，split() 方法被调用。split() 默认以空格为分隔符将字符串拆分成一个字符串列表。
    map(float, ...) 将字符串转换为浮点数              """
    for line in lines:
        column1, column2 = map(float, line.strip().split())
        data1.append(column1)
        data2.append(column2)

    with open(isAdmit_path, 'r') as file:
        lines = file.readlines()
    # float(line.strip())将经过处理的行转换为浮点数类型
    data = [float(line.strip()) for line in lines]
    return torch.tensor(data1), torch.tensor(data2), torch.tensor(data)


exam1, exam2, isAdmit = readFile(file_path, isAdmit_path)

# print(exam1)
# print(exam2)
# print(isAdmit)

admitted_points_x, admitted_points_y, rejected_points_x, rejected_points_y = [], [], [], []

for i in range(len(isAdmit)):
    if (isAdmit[i].item() == 1):
        admitted_points_x.append(exam1[i].item())
        admitted_points_y.append(exam2[i].item())
    else:
        rejected_points_x.append(exam1[i].item())
        rejected_points_y.append(exam2[i].item())
plt.figure()
plt.scatter(admitted_points_x, admitted_points_y, color='green', label='Admitted')
plt.scatter(rejected_points_x, rejected_points_y, color='red', label='Not Admitted')

plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')

# 定义参数：学习率、迭代次数
learning_rate = 0.0001
num_epochs = 1000
# 记录每次loss随着迭代次数的变化
loss_x, loss_y = [], []
# 将样本数据拼接成一个矩阵 先转换成列向量
data = torch.cat((exam1.view(-1, 1), exam2.view(-1, 1)), dim=1)  # 将两列向量拼成一个矩阵
ones = torch.ones(exam1.size(0), 1, dtype=torch.float32)  # 80*1 列向量
dataHat = torch.cat((ones, data), dim=1)

theta = torch.tensor([-0.1, 0, 0], dtype=torch.float32, requires_grad=True)

for epoch in range(num_epochs):
    # 随机选择一个样本
    random_index = torch.randint(0, dataHat.size(0), (1,))  # 行向量
    random_sample = dataHat[random_index]
    random_label = isAdmit[random_index]

    prediction = torch.sigmoid(torch.matmul(random_sample, theta.view(-1, 1)))
    # print(prediction.shape)

    loss = -(random_label * torch.log(prediction) + (1 - random_label) * torch.log(1 - prediction))
    loss_x.append(epoch + 1)
    loss_y.append(loss.item())

    # 使用自动微分计算梯度
    loss.backward()

    # 更新模型参数
    with torch.no_grad():
        theta -= learning_rate * theta.grad
        theta.grad.zero_()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'训练得到的模型参数：theta0 = {theta[0].item()},theta1 = {theta[1].item()},theta2 = {theta[2].item()}')

x = np.linspace(exam1.min(), exam1.max(), 100)
y = (-theta[0].item() - theta[1].item() * x) / theta[2].item()

plt.plot(x, y, color="black", linestyle='dashed', label='Decision Boundary')
plt.legend()
plt.show()

#图二 loss随梯度次数变化
plt.figure()
plt.plot(loss_x,loss_y)
plt.scatter(loss_x,loss_y,color='red')
plt.xlabel('iterations')
plt.ylabel('value of loss')
plt.show()