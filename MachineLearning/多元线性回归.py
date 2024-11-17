import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
import joblib

lin_reg = LinearRegression()
poly = PolynomialFeatures(degree=2)

csv_path = r"J:\23_06_lunwen\231017 只用尼泊尔境内数据\231017 aqi_pm2.5尼泊尔数据匹配.csv"

csv_data = pd.read_csv(csv_path)

pm = csv_data['PM2.5'].to_numpy().reshape(-1, 1)
aqi = csv_data['AQI'].to_numpy().reshape(-1, 1)

# 3次多项式拟合最好
poly.fit(aqi)
x = poly.transform(aqi)

lin_reg = LinearRegression()

lin_reg.fit(x, pm)
predict = lin_reg.predict(x)

joblib.dump(lin_reg, 'len_reg_china.m')

a = joblib.load('len_reg_china.m')
print(a.coef_)
print(a.intercept_)
exit(0)
aqi_country = r'"J:\23_06_lunwen\231017 只用尼泊尔境内数据\231017 17-19年TOA_AQI_尼泊尔.csv"'
data = pd.read_csv(aqi_country)
aqi = data['AQI'].to_numpy().reshape(-1, 1)
# x = poly.transform(aqi)

y = a.predict(x)
data['真实pm2.5'] = y
data.to_csv('J:\step3\MachineLearning\多项式计算pm2.5_二次_china.csv', encoding='gbk')

# print(lin_reg.coef_)
# plt.scatter(aqi, pm)
# plt.plot(np.sort(aqi), predict, color='r')
# plt.show()





