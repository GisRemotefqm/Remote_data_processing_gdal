from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
import os
from XXX项目.MachineLearning.日平均 import model_use
import numpy as np


df = pd.read_csv('H:\step3\多点数据汇总.csv')

# 'band1', 'band2', 'band3', 'band4'
df = df[['blh', 'r', 'sp', 't2m',
        'wd', 'ws', 'dem', 'band1', 'band2', 'band3', 'band4', 'pm2.5']]

print(df.head())
X = df.iloc[:, :-1]
ndvi = (X['band4'].to_numpy() - X['band3'].to_numpy()) / (X['band4'].to_numpy() + X['band3'].to_numpy())
ndvi = np.nan_to_num(ndvi, nan=0)
X['ndvi'] = ndvi
X = X.values.astype('float32')
# X = df.iloc[:, :-1].values.astype('float32')
y = df['pm2.5'].values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=3)

params = {'n_estimators': range(700, 1400),
          'max_depth': range(1, 20),
          }
gbr = GradientBoostingRegressor(learning_rate=0.001)

gbdt_search = RandomizedSearchCV(gbr, params, n_iter=30, cv=10)
gbdt_search.fit(X, y)
print(gbdt_search.best_params_)
gbdt_search.best_estimator_.fit(X, y)
y_train_pred = gbdt_search.best_estimator_.predict(X_train)
y_test_pred = gbdt_search.best_estimator_.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)

for train_index, test_index in kfold.split(X, y):

    # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
    this_train_x, this_train_y = X[train_index], y[train_index]  # 本组训练集
    this_test_x, this_test_y = X[test_index], y[test_index]  # 本组验证集
    # print(this_test_x, this_test_y)
    # 训练本组的数据，并计算准确率
    gbdt_search.best_estimator_.fit(this_train_x, this_train_y)
    prediction = gbdt_search.best_estimator_.predict(this_test_x)
    score = r2_score(this_test_y, prediction)
    print(score)  # 得到预测结果区间[0,1]


tif_path = r'H:\第一批数据去云表观\GF1_2020\尼泊尔'
blh_path = r'H:\尼泊尔\blh'
r_path = r'H:\尼泊尔\r'
sp_path = r'H:\尼泊尔\sp'
tp_path = r'H:\尼泊尔\tp'
wd_path = r'H:\尼泊尔\wd'
ws_path = r'H:\尼泊尔\ws'
t2m_path = r'H:\尼泊尔\t2m'
output = r'H:\机器学习测试文件\result'
machine_path = 'bag_model.m'

for temp in os.listdir(tif_path):

    if os.path.splitext(temp)[1] == '.tif':

        tif_name = os.path.join(tif_path, temp)
        date = temp.split('_')[4]
        blh_name = os.path.join(blh_path, date[:4] + '-' + date[4:6] + '-' + date[6:] + '_blh_resample.tif')
        r_name = os.path.join(r_path, date[:4] + '-' + date[4:6] + '-' + date[6:] + '_r_resample.tif')
        sp_name = os.path.join(sp_path, date[:4] + '-' + date[4:6] + '-' + date[6:] + '_sp_resample.tif')
        tp_name = os.path.join(tp_path, date[:4] + '-' + date[4:6] + '-' + date[6:] + '_tp_resample.tif')
        wd_name = os.path.join(wd_path, date[:4] + '-' + date[4:6] + '-' + date[6:] + '_wd_resample.tif')
        ws_name = os.path.join(ws_path, date[:4] + '-' + date[4:6] + '-' + date[6:] + '_ws_resample.tif')
        t2m_name = os.path.join(t2m_path, date[:4] + '-' + date[4:6] + '-' + date[6:] + '_t2m_resample.tif')
        outputpath = os.path.join(output, os.path.splitext(temp)[0] + '_machine_gdbt.tif')

        model_use.machine_tif(gbdt_search.best_estimator_, tif_name, blh_name, r_name, sp_name, tp_name, wd_name, ws_name, t2m_name, outputpath)

# plt.savefig('images/10_14.png', dpi=300)
plt.show()
joblib.dump(gbdt_search.best_estimator_, '日平均/日平均模型结果/xgboost模型/gbdt_model_mul.m')
