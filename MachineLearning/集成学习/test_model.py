# --coding:utf-8--
import numpy as np
import pandas as pd
import stacking_machine
import xgboost as xgb
import lightgbm as lgg
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import joblib
from sklearnex import patch_sklearn, unpatch_sklearn
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import LeaveOneOut

patch_sklearn()

df = pd.read_csv(r"C:\Users\fuqiming\Desktop\Landsat卫星表观反射率表\汇总匹配\日汇总+月汇总.csv", encoding='gbk')

df2 = df[['band3', 'band4']]
# 获得经纬度
data_date = df['date']
lon = df['lon']
lat = df['lat']

df = df[['blh', 'r', 'sp', 't2m', 'wd', 'ws', 'band1', 'band2', 'dem_no', 'pop', 'random_pm']]
X = df.iloc[:, :-1]
ndvi = (df2['band4'].to_numpy() - df2['band3'].to_numpy()) / (df2['band4'].to_numpy() + df2['band3'].to_numpy())
ndvi = np.nan_to_num(ndvi, nan=0)

X['ndvi'] = ndvi
X = X.values.astype('float32')
y = df['random_pm'].values.astype('float32')

# 训练的模型参数
xgb_dict = {'n_estimators': tuple(range(100, 700))}
xgb_dict1 = {'max_depth': tuple(range(1, 15)),
             'min_child_weight': tuple(range(1, 11, 1))}
xgb_dict2 = {'gamma': tuple([i/10.0 for i in range(0, 5)])}
xgb_dict3 = {'subsample': tuple([i/100.0 for i in range(75, 90, 2)]),
            'colsample_bytree': tuple([i/100.0 for i in range(75, 90, 2)])}
xgb_dict4 = {'learning_rate': (tuple([i * 0.01 for i in range(1, 11)]))}

lgbm_dict = {
    'n_estimators': tuple(range(100, 900)),
            }
lgbm_dict1 = {
    'max_depth': tuple(range(1, 10)),
    'num_leaves': tuple(range(1, 50))
}
lgbm_dict2 = {
    'min_child_samples': tuple(range(1, 22)),
    'min_child_weight': (0.001, 0.002)
              }
lgbm_dict3 = {
    'subsample': (0.8, 0.9, 1.0),
    'colsample_bytree': (0.8, 0.9, 1.0),
}
lgbm_dict4 = {
    'reg_alpha': (0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5),
    'reg_lambda': (0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5)
}
lgbm_dict5 = {
    'learning_rate': (tuple([i * 0.01 for i in range(1, 10)]))
}

forest_params_dict = {
    'n_estimators': tuple(range(100, 900)),
                      }
forest_params_dict1 = {
    'min_samples_split': tuple(range(1, 201))
}
forest_params_dict2 = {
    'max_depth': tuple(range(1, 15))
}
forest_params_dict3 = {
    'min_samples_leaf': tuple(range(10, 60))
}
forest_params_dict4 = {
    'max_features': tuple(range(3, 11))
}

et_dict = {
    'n_estimators': tuple(range(100, 900)),
    'max_depth': tuple(range(1, 15))
           }

XGB = joblib.load('./first_model/firest_XGB_month_mean.m')
ET = joblib.load('./first_model/firest_ET_month_mean.m')
Light = joblib.load('./first_model/firest_Light_month_mean.m')
RF = joblib.load('./first_model/firest_RF_month_mean.m')

# 第一层四个的顺序
models = [XGB, ET, Light, RF]
name = ['xgb', 'et', 'light', 'rf']

# 设置第二层的训练集
second_x, second_y = [], []
for i in range(len(models)):
    predict, target = stacking_machine.kfold_sample(models[i], X, y, data_date)
    second_x.append(predict)
    second_y.append(target)

second_x = np.array(second_x).T
second_y = np.array(second_y).T.mean(axis=1)
print(second_y.shape, second_x.shape)

print('第二层训练开始')
print('*' * 50)
print('*' * 50)

print('RF得到得结果为：')
RF = RandomForestRegressor()
RF = stacking_machine.search_bestparams(RF, forest_params_dict, second_x, second_y)
# RF = search_bestparams(RF, forest_params_dict1, X, y)
RF = stacking_machine.search_bestparams(RF, forest_params_dict2, second_x, second_y)
predict, target = stacking_machine.kfold_sample(RF, second_x, second_y, data_date)
result = pd.DataFrame(np.hstack((predict, target)).T)
result.to_csv('第一层两个LX_第二层RF.csv')

print('贝叶斯得到得结果为：')
clf = BayesianRidge(compute_score=True)
clf.fit(second_x, second_y)
predict, target = stacking_machine.kfold_sample(clf, second_x, second_y, data_date)
result = pd.DataFrame(np.hstack((predict, target)).T)
result.to_csv('第一层两个LX_第二层贝叶斯.csv')

print('ET得到得结果为：')
ET = ExtraTreesRegressor(criterion='squared_error', n_jobs=-1, random_state=3)
ET = stacking_machine.search_bestparams(ET, et_dict, second_x, second_y)
predict, target = stacking_machine.kfold_sample(ET, second_x, second_y, data_date)
result = pd.DataFrame(np.hstack((predict, target)).T)
result.to_csv('第一层两个LX_第二层ET.csv')

print('XGB得到的结果为：')
XGB = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=300)
XGB = stacking_machine.search_bestparams(XGB, xgb_dict, second_x, second_y)
XGB = stacking_machine.search_bestparams(XGB, xgb_dict1, second_x, second_y)
XGB = stacking_machine.search_bestparams(XGB, xgb_dict2, second_x, second_y)
XGB = stacking_machine.search_bestparams(XGB, xgb_dict3, second_x, second_y)
predict, target = stacking_machine.kfold_sample(XGB, second_x, second_y, data_date)
result = pd.DataFrame(np.hstack((predict, target)).T)
result.to_csv('第一层两个LX_第二层XGB.csv')

print('Light得到得结果为')
Light = lgg.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=500)
Light = stacking_machine.search_bestparams(Light, lgbm_dict, second_x, second_y)
Light = stacking_machine.search_bestparams(Light, lgbm_dict5, second_x, second_y)
predict, target = stacking_machine.kfold_sample(Light, second_x, second_y, data_date)
result = pd.DataFrame(np.hstack((predict, target)).T)
result.to_csv('第一层两个LX_第二层Light.csv')


joblib.dump(RF, './第一层两个模型_第二层模型结果/LX_RF.m')
joblib.dump(ET, './第一层两个模型_第二层模型结果/LX_ET.m')
joblib.dump(Light, './第一层两个模型_第二层模型结果/LX_Light.m')
joblib.dump(XGB, './第一层两个模型_第二层模型结果/LX_XGB.m')