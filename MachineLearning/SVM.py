from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import RepeatedKFold


'''
X：特征向量
y：样本的目标值
trn_x：训练集的特征
val_x：测试集的特征
trn_y：训练集的目标值
val_y：测试集的目标值
'''
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
'''
kernel参数可选:{'sigmod','rbf','sigmoid'...}
'''
svr = SVR(degree=3, coef0=0.0,
		tol = 0.00001,  epsilon = 0.01, cache_size = 200,
		verbose = False, max_iter = -1)

svr_param_dict = {
                  'degree': range(3, 5000),
                  'coef0': range(0, 200),
                  'epsilon': [i * 0.001 for i in range(1000)]
                  }

svr_search = RandomizedSearchCV(svr, svr_param_dict, n_iter=200, cv=10)
svr_search.fit(X, y)
print(svr_search.best_params_)
svr_search.best_estimator_.fit(X_train, y_train)
y_train_pred = svr_search.best_estimator_.predict(X_train)
y_test_pred = svr_search.best_estimator_.predict(X_test)

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
joblib.dump(svr_search.best_estimator_, 'svr_model.m')


