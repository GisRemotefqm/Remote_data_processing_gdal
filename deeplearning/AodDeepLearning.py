# --coding:utf-8--
import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from sklearn.metrics import r2_score
import joblib
from sklearnex import patch_sklearn, unpatch_sklearn
from sklearn.model_selection import RepeatedKFold
import lightgbm as lgg
patch_sklearn()


def search_bestparams(model, dict, X1, y1, model_outputPath, model_outputName, csvPath):

    # cv_scores = cross_val_score(model_search.best_estimator_, X1, y1, cv=10, scoring='r2')

    kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=3)
    cv_target, cv_predict = np.array([]), np.array([])
    flag = 0
    model_search = TuneSearchCV(model, dict, cv=10, use_gpu=True,
                                search_optimization="bayesian", max_iters=5000, n_jobs=-1)
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


if __name__ == '__main__':

    df = pd.read_csv(r"J:\AODinversion\匹配数据\TOA-AOD-qxNoResampleAddMonth16-20.csv", encoding='gbk')
    data_date = df['date']
    lon = df['lon']
    lat = df['lat']

    df = df[['B2', 'B3', 'B4', 'B5', 'tco', 'tcwv', 'NDVI', 'DEM', 'AOD_550nm']]

    X = df.iloc[:, :-1]

    X = X.values.astype('float32')
    y = df['AOD_550nm'].values.astype('float32')
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=3)

    # 使用的模型
    dnn = MLPRegressor(hidden_layer_sizes=(4000, 3000, 2000, 2000), activation='relu',
                       learning_rate_init=0.001, max_iter=5000)
    dnn.fit(train_x, train_y)
    predict = dnn.predict(test_x)
    print(r2_score(test_y, predict))

    # dnn.fit(X, y)
    xgb_dict = {
        # 'hidden_layer_sizes': ((50, 50), (100, 100), (150, 150), (200, 200),
        #                                (50, 50, 50), (100, 100, 100), (150, 150, 150),
        #                                (200, 200, 200)),
        'solver': ('sgd', 'adam'),
        'learning_rate': ('constant', 'invscaling', 'adaptive'),
        'max_iter': tuple(range(3000, 10000))
                }

    XGB = search_bestparams(dnn, xgb_dict, X, y, r'J:\AODinversion\2024大论文\DNNModel',
                            'thesis_dnn_model',
                            r'J:\AODinversion\2024大论文\CV\thesis_dnn_model.csv')

