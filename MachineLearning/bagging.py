from sklearn.ensemble import BaggingRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import RepeatedKFold

df = pd.read_csv('H:\step3\第一次匹配\数据汇总.csv')
# 'band1', 'band2', 'band3', 'band4'
df = df[['blh', 'r', 'sp', 't2m',
        'wd', 'ws', 'band1', 'band2', 'band3', 'band4', 'pm2.5']]

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
    X, y, test_size=0.1, random_state=1)

params = {'n_estimators': range(200, 700)}

clf = BaggingRegressor(n_estimators=165)


bag_search = RandomizedSearchCV(clf, params, n_iter=100, cv=10)
bag_search.fit(X, y)
print(bag_search.best_params_)
bag_search.best_estimator_.fit(X_train, y_train)
y_train_pred = bag_search.best_estimator_.predict(X_train)
y_test_pred = bag_search.best_estimator_.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

plt.scatter(y_train_pred,
            y_train_pred - y_train,
            c='steelblue',
            edgecolor='white',
            marker='o',
            s=35,
            alpha=0.9,
            label='training data')
plt.scatter(y_test_pred,
            y_test_pred - y_test,
            c='limegreen',
            edgecolor='white',
            marker='s',
            s=35,
            alpha=0.9,
            label='test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('images/10_14.png', dpi=300)
plt.show()
joblib.dump(bag_search.best_estimator_, 'bag_model.m')


kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)

for train_index, test_index in kfold.split(X, y):

    # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
    this_train_x, this_train_y = X[train_index], y[train_index]  # 本组训练集
    this_test_x, this_test_y = X[test_index], y[test_index]  # 本组验证集
    # print(this_test_x, this_test_y)
    # 训练本组的数据，并计算准确率
    bag_search.best_estimator_.fit(this_train_x, this_train_y)
    prediction = bag_search.best_estimator_.predict(this_test_x)
    score = r2_score(this_test_y, prediction)
    print(score)  # 得到预测结果区间[0,1]