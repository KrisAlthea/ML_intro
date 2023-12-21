import numpy as np

# 给出历史数据
x = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013])
y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900])

# 数据正则化，使得特征在同一数量级，加速收敛
x_norm = (x - np.mean(x)) / np.std(x)

# 初始化参数
theta_0 = 0
theta_1 = 0

# 设置学习率和迭代次数
alpha = 0.01
iters = 3500  # 3200左右收敛


# 均方误差损失函数
def compute_cost(x, y, theta_0, theta_1):
    return np.sum((theta_0 + theta_1 * x - y) ** 2) / (2 * len(x))


# 梯度下降算法求解参数theta_0, theta_1
def gradient_descent(x, y, theta_0, theta_1, alpha, iters):
    m = len(x)
    for i in range(iters):
        temp_theta_0 = theta_0 - alpha * (np.sum(theta_0 + theta_1 * x - y) / m)
        temp_theta_1 = theta_1 - alpha * (np.sum((theta_0 + theta_1 * x - y) * x) / m)
        theta_0 = temp_theta_0
        theta_1 = temp_theta_1
        cost = compute_cost(x, y, theta_0, theta_1)
        print(f"Iteration {i + 1}: cost = {cost}")
    return theta_0, theta_1


# 闭式解方法求解参数theta
def closed_form_solution(x, y):
    X = np.column_stack((np.ones_like(x), x))  # 添加一列常数项
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


# 使用梯度下降法
theta_0, theta_1 = gradient_descent(x_norm, y, theta_0, theta_1, alpha, iters)
print(f"梯度下降算法线性关系模型： y = {theta_0:.4f} + {theta_1:.4f} * x_norm")

# 使用闭式解方法
theta_closed_form = closed_form_solution(x_norm, y)
print(f"LMS的闭式解线性关系模型： y = {theta_closed_form[0]:.4f} + {theta_closed_form[1]:.4f} * x_norm")

# 对2014年的房价进行预测（使用梯度下降法得到的参数）
x_predict = (2014 - np.mean(x)) / np.std(x)
y_predict_gradient_descent = theta_0 + theta_1 * x_predict
print(f"梯度下降算法预测 2014 年南京的平均房价：{y_predict_gradient_descent:.4f}")

# 对2014年的房价进行预测（使用闭式解方法得到的参数）
y_predict_closed_form = theta_closed_form[0] + theta_closed_form[1] * x_predict
print(f"LMS的闭式解预测 2014 年南京的平均房价：{y_predict_closed_form:.4f}")
