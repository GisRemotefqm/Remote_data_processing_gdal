# --coding:utf-8--
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgg
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
import joblib
from sklearnex import patch_sklearn
from tune_sklearn import TuneSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge
patch_sklearn()
from GetData import GetImageData

def search_bestparams(model, dict, X1, y1, model_outputPath, model_outputName, csvPath):

    # cv_scores = cross_val_score(model_search.best_estimator_, X1, y1, cv=10, scoring='r2')

    kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=3)
    cv_target, cv_predict = np.array([]), np.array([])
    flag = 0
    model_search = TuneSearchCV(model, dict, cv=10, use_gpu=True,
                                search_optimization="bayesian", max_iters=1, n_jobs=-1)
    model_search.fit(X1, y1)
    for train_index, test_index in kfold.split(X1, y1):

        train_con, val_con = X1[train_index], X1[test_index]
        train_tar, val_tar = y1[train_index], y1[test_index]


        print(model_search.best_params_)
        # model_search.best_estimator_.fit(X1, y1)
        model = model_search.best_estimator_
        # print(this_test_x, this_test_y)
        # 训练本组的数据，并计算准确率
        # model.fit(train_con, train_tar)
        model_search.best_estimator_.fit(train_con, train_tar)
        prediction = model_search.best_estimator_.predict(val_con)
        score = r2_score(val_tar, prediction)
        print(score)  # 得到预测结果区间[0,1]
        cv_target = np.concatenate((cv_target, val_tar))
        cv_predict = np.concatenate((cv_predict, prediction))
        joblib.dump(model_search.best_estimator_, os.path.join(model_outputPath, model_outputName + '_' + str(flag) + '.m'))
        flag += 1
    print('final cv: ', r2_score(cv_target, cv_predict))
    print('-' * 50)
    result = pd.DataFrame(np.vstack((cv_target, cv_predict)).T, columns=['target', 'predict'])
    result.to_csv(csvPath)


    return model_search.best_estimator_


def search_bestparams1(model, dict, X1, y1, data_date, lon, lat):
    model_search = TuneSearchCV(model, dict, cv=4, use_gpu=True,
                                 search_optimization="bayesian", max_iters=1, n_jobs=-1)


    kfold = KFold(n_splits=5, shuffle=True)
    cv_target, cv_predict = np.array([]), np.array([])
    for train_index, test_index in kfold.split(X1, y1):

        train_con, val_con = X1[train_index], X1[test_index]
        train_tar, val_tar = y1[train_index], y1[test_index]
        # print(this_test_x, this_test_y)
        # 训练本组的数据，并计算准确率
        model_search.fit(train_con, train_tar)
        # model_search.best_estimator_.fit(train_con, train_tar)
        prediction = model_search.best_estimator_.predict(val_con)
        score = r2_score(val_tar, prediction)
        print(score)  # 得到预测结果区间[0,1]

        cv_target = np.concatenate((cv_target, val_tar))
        cv_predict = np.concatenate((cv_predict, prediction))

    print('final cv: ', r2_score(cv_target, cv_predict))
    print('-' * 50)
    kfold_sample(model_search.best_estimator_, X1, y1, data_date, lon, lat)
    return model_search.best_estimator_


def kfold_sample(model, X, y, data_date, lon, lat, model_outputPath, model_outputName, csvPath):
    kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=3)
    cv_target, cv_predict = np.array([]), np.array([])
    cv_date, cv_lat, cv_lon = np.array([]), np.array([]), np.array([])
    flag = 0
    for train_index, test_index in kfold.split(X, y):
        train_date, val_date = data_date[train_index], data_date[test_index]
        train_lat, val_lat = lat[train_index], lat[test_index]
        train_lon, val_lon = lon[train_index], lon[test_index]

        # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
        # this_train_x, this_train_y = X[train_index], y[train_index]  # 本组训练集
        # this_test_x, this_test_y = X[test_index], y[test_index]  # 本组验证集
        train_con, val_con = X[train_index], X[test_index]
        train_tar, val_tar = y[train_index], y[test_index]
        # print(this_test_x, this_test_y)
        # 训练本组的数据，并计算准确率
        model.fit(train_con, train_tar)
        prediction = model.predict(val_con)
        score = r2_score(val_tar, prediction)
        print(score)  # 得到预测结果区间[0,1]

        cv_target = np.concatenate((cv_target, val_tar))
        cv_predict = np.concatenate((cv_predict, prediction))

        cv_date = np.concatenate((cv_date, val_date))
        cv_lat = np.concatenate((cv_lat, val_lat))
        cv_lon = np.concatenate((cv_lon, val_lon))
        if model_outputPath == None:
            print('不需要保存')
        else:
            joblib.dump(model, os.path.join(model_outputPath, model_outputName + '_' + str(flag) + '.m'))
        flag += 1
    print('final cv: ', r2_score(cv_target, cv_predict))
    if csvPath == None:
        print('pass')
    else:
        result = pd.DataFrame(np.vstack((cv_target, cv_predict)).T, columns=['target', 'predict'])
        result.to_csv(csvPath, index=False)
    return cv_predict, cv_target


def kfold_sample2(model, X, y, data_date, lon, lat):
    kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=3)
    cv_target, cv_predict = np.array([]), np.array([])
    cv_date, cv_lat, cv_lon = np.array([]), np.array([]), np.array([])
    flag = 0
    for train_index, test_index in kfold.split(X, y):
        train_date, val_date = data_date[train_index], data_date[test_index]
        train_lat, val_lat = lat[train_index], lat[test_index]
        train_lon, val_lon = lon[train_index], lon[test_index]

        # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
        # this_train_x, this_train_y = X[train_index], y[train_index]  # 本组训练集
        # this_test_x, this_test_y = X[test_index], y[test_index]  # 本组验证集
        train_con, val_con = X[train_index], X[test_index]
        train_tar, val_tar = y[train_index], y[test_index]
        # print(this_test_x, this_test_y)
        # 训练本组的数据，并计算准确率
        model.fit(train_con, train_tar)
        prediction = model.predict(val_con)
        score = r2_score(val_tar, prediction)
        print(score)  # 得到预测结果区间[0,1]

        cv_target = np.concatenate((cv_target, val_tar))
        cv_predict = np.concatenate((cv_predict, prediction))

        cv_date = np.concatenate((cv_date, val_date))
        cv_lat = np.concatenate((cv_lat, val_lat))
        cv_lon = np.concatenate((cv_lon, val_lon))

    return cv_predict, cv_target


def second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat):
    # 设置第二层的训练集
    second_x, second_y = [], []
    for i in range(len(models)):
        predict, target = kfold_sample2(models[i], X, y, data_date, lon, lat)
        second_x.append(predict)
        second_y.append(target)

    second_x = np.array(second_x).T
    second_y = np.array(second_y).T.mean(axis=1)
    print(second_y.shape, second_x.shape)

    print('第二层训练开始')
    print('*' * 50)
    print('*' * 50)

    print('RF得到得结果为：')
    RF = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=3, n_jobs=-1)
    RF = search_bestparams(RF, forest_params_dict, second_x, second_y, os.path.dirname(model_paths[1]), os.path.basename(model_paths[1])[:-2], csvPaths[0])
    # RF = search_bestparams(RF, forest_params_dict1, X, y)
    # RF = search_bestparams(RF, forest_params_dict2, second_x, second_y)
    predict, target = kfold_sample(RF, second_x, second_y, data_date, lon, lat, os.path.dirname(model_paths[1]), os.path.basename(model_paths[1])[:-2], csvPaths[0])
    result = pd.DataFrame(np.vstack((target, predict)).T, columns=['target', 'predict'])
    result.to_csv(csvPaths[0], index=False)

    # RF = search_bestparams(RF, forest_params_dict3, X, y)
    # RF = search_bestparams(RF, forest_params_dict4, X, y)
    # joblib.dump(RF, 'second_RF.m')

    # print('贝叶斯得到得结果为：')
    # clf = BayesianRidge(compute_score=True)
    # clf.fit(second_x, second_y)
    # predict, target = kfold_sample(clf, second_x, second_y, data_date)
    # result = pd.DataFrame(np.hstack((predict, target)).T)
    # result.to_csv(csvPaths[1])

    print('ET得到得结果为：')
    ET = ExtraTreesRegressor(criterion='squared_error', n_jobs=-1, random_state=3)
    ET = search_bestparams(ET, et_dict, second_x, second_y, os.path.dirname(model_paths[2]), os.path.basename(model_paths[2])[:-2], csvPaths[2])
    predict, target = kfold_sample(ET, second_x, second_y, data_date, lon, lat, os.path.dirname(model_paths[2]), os.path.basename(model_paths[2])[:-2], csvPaths[2])
    # result = pd.DataFrame(np.vstack((target, predict)).T, columns=['target', 'predict'])
    # result.to_csv(csvPaths[2], index=False)

    print('XGB得到的结果为：')
    XGB = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=300, n_jobs=-1)
    XGB = search_bestparams(XGB, xgb_dict, second_x, second_y, os.path.dirname(model_paths[0]), os.path.basename(model_paths[0])[:-2], csvPaths[3])

    predict, target = kfold_sample(XGB, second_x, second_y, data_date, lon, lat, os.path.dirname(model_paths[0]), os.path.basename(model_paths[0])[:-2], csvPaths[3])
    # result = pd.DataFrame(np.vstack((target, predict)).T, columns=['target', 'predict'])
    # result.to_csv(csvPaths[3], index=False)

    print('Light得到的结果为：')
    Light = lgg.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=500, n_jobs=-1)
    Light = search_bestparams(Light, lgbm_dict, X, y, os.path.dirname(model_paths[3]), os.path.basename(model_paths[3])[:-2], csvPaths[4])
    predict, target = kfold_sample(Light, second_x, second_y, data_date, lon, lat, os.path.dirname(model_paths[3]), os.path.basename(model_paths[3])[:-2], csvPaths[4])
    # result = pd.DataFrame(np.vstack((target, predict)).T, columns=['target', 'predict'])
    # result.to_csv(csvPaths[4], index=False)


    print('LinearRegressgion得到得结果为')
    linear = LinearRegression()
    predict, target = kfold_sample(linear, second_x, second_y, data_date, lon, lat, os.path.dirname(model_paths[4]), os.path.basename(model_paths[4])[:-2], csvPaths[5])
    # result = pd.DataFrame(np.vstack((target, predict)).T, columns=['target', 'predict'])
    # result.to_csv(csvPaths[5], index=False)

    print('ridge结果为：')
    ridge = Ridge(alpha=1.0)
    predict, target = kfold_sample(ridge, second_x, second_y, data_date, lon, lat, os.path.dirname(model_paths[5]), os.path.basename(model_paths[5])[:-2], csvPaths[6])
    # result = pd.DataFrame(np.vstack((target, predict)).T, columns=['target', 'predict'])
    # result.to_csv(csvPaths[6], index=False)


    # joblib.dump(XGB, model_paths[0])
    # joblib.dump(RF, model_paths[1])
    # joblib.dump(ET, model_paths[2])
    # joblib.dump(Light, model_paths[3])
    # joblib.dump(linear, model_paths[4])
    # joblib.dump(ridge, model_paths[5])


if __name__ == '__main__':

    XGB = joblib.load(r"J:\AODinversion\2024大论文\ML\xgb\XGB_AOD-pm_0.m")
    ET = joblib.load(r"J:\AODinversion\2024大论文\ML\light\Light_AOD-pm_0.m")
    print(ET.get_params())
    print(XGB.get_params())
    exit(0)
    df = pd.read_csv(r"F:\研究生毕业论文AOD提取数据\真实PM2.5与AOD数据匹配_unique.csv")
    # df = pd.read_csv(r"J:\AODinversion\匹配数据\TOA-AOD-qxNoResampleAddMonth16-20.csv")
    # df2 = df[['B4', 'B5']]
    # 获得经纬度
    data_date = df['date']
    lon = df['lon']
    lat = df['lat']

    # df = df[['blh', 'r', 'sp', 't2m', 'wd', 'ws', 'band1', 'band2', 'DEM', 'pop', 'PM2.5']]
    df = df[['AOD', 'blh', 'r', 'sp', 't2m', 'wd', 'ws', 'PM2.5']];
    X = df.iloc[:, :-1]
    # ndvi = (df2['B5'].to_numpy() - df2['B4'].to_numpy()) / (df2['B5'].to_numpy() + df2['B4'].to_numpy())
    # ndvi = np.nan_to_num(ndvi, nan=0)
    #
    # X['ndvi'] = ndvi
    X = X.values.astype('float32')
    y = df['PM2.5'].values.astype('float32')

    # 使用的模型
    XGB = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=300, n_jobs=-1)
    Light = lgg.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=500, n_jobs=-1)
    RF = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=3, n_jobs=-1)
    ET = ExtraTreesRegressor(criterion='squared_error', n_jobs=-1, random_state=3)

    xgb_dict = {'n_estimators': tuple(range(100, 700)),
                'max_depth': tuple(range(1, 15)),
                'min_child_weight': tuple(range(1, 11, 1)),
                'gamma': tuple([i / 10.0 for i in range(0, 5)]),
                'subsample': tuple([i / 100.0 for i in range(75, 90, 2)]),
                'learning_rate': tuple([0.05, 0.01, 0.25, 0.5]),
                'colsample_bytree': tuple([i / 100.0 for i in range(75, 90, 2)])
                }
    xgb_dict4 = {
        'learning_rate': (tuple([i * 0.01 for i in range(1, 11)]))}
    lgbm_dict = {
        'n_estimators': tuple(range(100, 900)),
        'max_depth': tuple(range(1, 15)),
    }
    forest_params_dict = {
        'n_estimators': tuple(range(100, 900)),
        'max_depth': tuple(range(1, 15))
    }
    et_dict = {
        'n_estimators': tuple(range(100, 900)),
        'max_depth': tuple(range(1, 15))
    }


    XGB =search_bestparams(XGB, xgb_dict, X, y, r'J:\AODinversion\2024大论文\ML\xgb',
                           'XGB_AOD-pm', r'F:\研究生毕业论文AOD提取数据\model\xgb\XGB_AOD-pm_predict.csv')

    Light = search_bestparams(Light, lgbm_dict, X, y, r'J:\AODinversion\2024大论文\ML\light',
                           'Light_AOD-pm',r'F:\研究生毕业论文AOD提取数据\model\light\Light_AOD-pm_predict.csv')

    ET = search_bestparams(ET, et_dict, X, y, r'J:\AODinversion\2024大论文\ML\et',
                           'ET_AOD-pm', r'F:\研究生毕业论文AOD提取数据\model\et\ET_AOD-pm_predict.csv')

    RF = search_bestparams(RF, forest_params_dict, X, y, r'J:\AODinversion\2024大论文\ML\forest',
                           'RF_AOD-pm', r'F:\研究生毕业论文AOD提取数据\model\forest\RF_AOD-pm_predict.csv')

    # models = [ET, XGB]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\EX_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EX_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EX_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EX_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EX_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EX_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EX_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\EX_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EX_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EX_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EX_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EX_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EX_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)
    #
    # models = [ET, Light]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\EL_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EL_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EL_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EL_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EL_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EL_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EL_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\EL_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EL_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EL_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EL_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EL_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EL_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)
    #
    # models = [ET, RF]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\ER_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\ER_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\ER_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\ER_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\ER_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\ER_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\ER_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\ER_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\ER_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\ER_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\ER_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\ER_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\ER_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)

    models = [XGB, Light]
    csvPaths = [r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_RF_AOD-pm.csv',
                r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_贝叶斯_AOD-pm.csv',
                r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_ET_AOD-pm.csv',
                r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_XGB_AOD-pm.csv',
                r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_Light_AOD-pm.csv',
                r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_线性_AOD-pm.csv',
                r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_ridge_AOD-pm.csv'
                ]
    model_paths = [r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_XGB_AOD-pm.m',
                   r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_RF_AOD-pm.m',
                   r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_ET_AOD-pm.m',
                   r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_Light_AOD-pm.m',
                   r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_线性_AOD-pm.m',
                   r'F:\研究生毕业论文AOD提取数据\model\stacking\XL_ridge_AOD-pm.m'
                   ]
    second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)

    # models = [XGB, RF]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\XR_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XR_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XR_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XR_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XR_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XR_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XR_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\XR_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XR_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XR_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XR_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XR_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XR_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)

    # models = [Light, RF]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\LR_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\LR_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\LR_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\LR_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\LR_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\LR_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\LR_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\LR_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\LR_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\LR_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\LR_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\LR_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\LR_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)
    #
    # models = [ET, XGB, RF]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\EXR_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXR_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXR_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXR_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXR_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXR_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXR_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\EXR_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXR_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXR_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXR_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXR_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXR_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)
    #
    # models = [ET, XGB, Light]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\EXL_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXL_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXL_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXL_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXL_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXL_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXL_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\EXL_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXL_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXL_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXL_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXL_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXL_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)
    #
    # models = [XGB, RF, Light]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\XRL_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XRL_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XRL_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XRL_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XRL_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XRL_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\XRL_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\XRL_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XRL_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XRL_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XRL_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XRL_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\XRL_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)
    #
    # models = [ET, XGB, RF, Light]
    # csvPaths = [r'J:\AODinversion\2024大论文\ML\stacking\EXRL_RF_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXRL_贝叶斯_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXRL_ET_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXRL_XGB_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXRL_Light_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXRL_线性_AOD.csv',
    #             r'J:\AODinversion\2024大论文\ML\stacking\EXRL_ridge_AOD.csv'
    #             ]
    # model_paths = [r'J:\AODinversion\2024大论文\ML\stacking\EXRL_XGB_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\ERL_RF_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXRL_ET_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXRL_Light_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXRL_线性_AOD.m',
    #                r'J:\AODinversion\2024大论文\ML\stacking\EXRL_ridge_AOD.m'
    #                ]
    # second_model_train(models, X, y, csvPaths, model_paths, data_date, lon, lat)