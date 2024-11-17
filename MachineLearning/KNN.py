from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv('H:\step3\数据汇总.csv')

df = df[['blh', 'r', 'sp', 't2m', 'tp',
        'wd', 'ws', 'band1', 'band2',
         'band3', 'band4', 'pm2.5']]
print(df.head())
X = df.iloc[:, :-1].values
y = df['pm2.5'].values

from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
standardscaler.fit(X)
X= standardscaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

clf = KNeighborsRegressor()

knn_param_dict = {'n_neighbors': range(0, 200),
                  'leaf_size': range(30, 300),
                  }

knn_search = RandomizedSearchCV(clf, knn_param_dict, n_iter=100, cv=3)
knn_search.fit(X, y)

print(knn_search.best_params_)
knn_search.best_estimator_.fit(X_train, y_train)
y_train_pred = knn_search.best_estimator_.predict(X_train)
y_test_pred = knn_search.best_estimator_.predict(X_test)

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
joblib.dump(knn_search.best_estimator_, 'knn_model.m')

