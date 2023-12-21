import numpy as np
import matplotlib.pyplot as plt

# 提供的数据
X_values = np.array([[1, 2000], [1, 2001], [1, 2002], [1, 2003], [1, 2004], [1, 2005], [1, 2006], [1, 2007], [1, 2008],
                     [1, 2009], [1, 2010], [1, 2011], [1, 2012], [1, 2013]])
y_values = np.array(
    [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900])

# 计算 LMS 闭式解
weights_lms = np.linalg.inv(X_values.T @ X_values) @ X_values.T @ y_values

# 预测值
y_pred = X_values @ weights_lms

# 添加2014年数据点
year_2014 = 2014
predicted_price_2014 = weights_lms[0] + weights_lms[1] * year_2014

# 绘制数据点
plt.scatter(X_values[:, 1], y_values, label='Data Points')

# 绘制拟合曲线
plt.plot(X_values[:, 1], y_pred, color='red', label='Linear Regression')

# 绘制预测的2014年数据点
plt.scatter(year_2014, predicted_price_2014, color='green', label='Predicted 2014')

plt.plot([X_values[-1, 1], year_2014], [y_pred[-1], predicted_price_2014], linestyle='--', color='green')

# 设置x轴范围
plt.xlim(2000, 2016)

# 设置y轴范围
plt.ylim(0, 15)

# 添加标签和标题
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Linear Regression Fit')

# 显示图例
plt.legend()

# 显示图形
plt.show()
