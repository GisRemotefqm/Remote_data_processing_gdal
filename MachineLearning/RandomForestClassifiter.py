# --coding:utf-8--
import glob
import time
from sklearn.ensemble import ExtraTreesClassifier
import xgboost
import lightgbm
import joblib
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn
import os
from DataMath import ReadWrite_h5 as rw
from sklearn.metrics import accuracy_score
import osgeo.osr as osr
import sys


def readTrainData(inputPath_dict):
    data = []
    for key in inputPath_dict.keys():
        paths = inputPath_dict[key]
        for path in paths:

            tifDataset = rw.get_tifDataset(path)
            imgx, imgy = rw.get_RasterXY(tifDataset)
            tifArr = rw.get_RasterArr(tifDataset, imgx, imgy)
            # print(tifArr.shape)
            # print(tifArr[0].astype(np.float32).shape, tifArr[0].astype(np.float32).reshape(-1).shape)
            # exit(0)
            v1 = tifArr[0].astype(np.float32).reshape(-1)
            v2 = tifArr[1].astype(np.float32).reshape(-1)
            v3 = tifArr[2].astype(np.float32).reshape(-1)
            v4 = tifArr[3].astype(np.float32).reshape(-1)
            v1 = v1[~np.isnan(v4)]
            v2 = v2[~np.isnan(v4)]
            v3 = v3[~np.isnan(v4)]
            v4 = v4[~np.isnan(v4)]

            v1 = v1[v4 != 0.0]
            v2 = v2[v4 != 0.0]
            v3 = v3[v4 != 0.0]
            v4 = v4[v4 != 0.0]
            ndvi = (v4 - v3) / (v4 + v3)
            label = np.zeros(v1.shape[0])
            print(key, label.shape[0])
            label[label == 0] = key
            print(key)
            print('-' * 50)
            data.append(np.vstack((v1, v2, v3, v4, ndvi, label)).T)

    data = np.concatenate(data, axis=0)
    print(data[:, :-4].shape)
    train_x, test_x, train_y, test_y = train_test_split(data[:, :5], data[:, 5], test_size=0.2, random_state=5)
    return train_x, test_x, train_y, test_y


def find_and_compute_areas(matrix, unit_area=0.8):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    m, n = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    regions = []
    stack = []

    def dfs(i, j, label):
        count = 0
        print('find_and_compute_areas', 1111)
        stack.append((i, j))
        while stack:
            i, j = stack.pop()

            if i < 0 or i >= m or j < 0 or j >= n or matrix[i, j] == 0 or visited[i, j]:
                continue

            visited[i, j] = True

            if matrix[i, j] == -2:
                visited[i, j] = False
                matrix[i, j] = 1

            if len(regions) <= label:
                regions.append(set())

            regions[label].add((i, j))
            count += 1

            # 添加相邻的1到栈中
            stack.extend([(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)])
        return count

    label = 0
    areas = []
    for i in range(m):
        for j in range(n):
            if matrix[i, j] == 1 and not visited[i, j]:
                count = dfs(i, j, label)
                area = count * unit_area
                areas.append(area)
                label += 1
    print('find_and_compute_areas', len(regions))
    return matrix, areas


def train(train_x, test_x, train_y, test_y):

    forest = RandomForestClassifier(n_estimators=200, max_depth=12, criterion='entropy', bootstrap=True)
    xgb = xgboost.XGBClassifier(objective='multi:softmax', booster='gbtree', max_depth=6, learning_rate=0.05, n_estimators=300, n_jobs=-1)
    light = lightgbm.LGBMClassifier(objective='multiclass', learning_rate=0.05, n_estimators=500, n_jobs=-1)
    et = ExtraTreesClassifier(criterion='gini', n_jobs=-1, random_state=3)
    forest_params_dict = {
        'n_estimators': tuple(range(100, 900)),
        'min_weight_fraction_leaf': tuple([i * 0.001 for i in range(0, 101)]),
        'max_depth': tuple(range(1, 15)),
        'min_samples_leaf': tuple([i * 0.1 for i in range(1, 5)]),
        'min_samples_split': tuple([i * 0.1 for i in range(1, 10)]),
        }

    lgbm_dict = {
        'n_estimators': tuple(range(100, 900)),
        'max_depth': tuple(range(1, 15)),
    }

    et_dict = {
        'n_estimators': tuple(range(100, 900)),
        'max_depth': tuple(range(1, 15))
    }

    xgb_dict = {'n_estimators': tuple(range(100, 700)),
                'max_depth': tuple(range(1, 15)),
                'min_child_weight': tuple(range(1, 11, 1)),
                'gamma': tuple([i / 10.0 for i in range(0, 5)]),
                'subsample': tuple([i / 100.0 for i in range(75, 90, 2)]),
                'learning_rate': tuple([0.05, 0.01, 0.25, 0.5]),
                'colsample_bytree': tuple([i / 100.0 for i in range(75, 90, 2)])
                }

    random_search = TuneSearchCV(forest, forest_params_dict, cv=10, use_gpu=True,
                                 search_optimization="bayesian", max_iters=100, n_jobs=-1)

    lgbm_search = TuneSearchCV(light, lgbm_dict, cv=10, use_gpu=True,
                                 search_optimization="bayesian", max_iters=100, n_jobs=-1)

    et_search = TuneSearchCV(et, et_dict, cv=10, use_gpu=True,
                                 search_optimization="bayesian", max_iters=100, n_jobs=-1)

    xgb_search = TuneSearchCV(xgb, xgb_dict, cv=10, use_gpu=True,
                                 search_optimization="bayesian", max_iters=100, n_jobs=-1)

    print(train_x.shape, train_y.shape)
    random_search.fit(train_x, train_y)
    forest = random_search.best_estimator_
    predict_y = forest.predict(test_x)
    score = accuracy_score(test_y, predict_y)
    print(score)
    print('-' * 50)

    et_search.fit(train_x, train_y)
    et = et_search.best_estimator_
    predict_y = et.predict(test_x)
    score = accuracy_score(test_y, predict_y)
    print(score)
    print('-' * 50)

    xgb_search.fit(train_x, train_y)
    xgb = xgb_search.best_estimator_
    predict_y = xgb.predict(test_x)
    score = accuracy_score(test_y, predict_y)
    print(score)
    print('-' * 50)

    lgbm_search.fit(train_x, train_y)
    light = lgbm_search.best_estimator_
    predict_y = light.predict(test_x)
    score = accuracy_score(test_y, predict_y)
    print(score)
    print('-' * 50)

    return forest, et, xgb, light


def ModelPredict(model, inputPath, outPath):
    start_time = time.time()
    tifDataset = rw.get_tifDataset(inputPath)
    project, transform = rw.get_GeoInformation(tifDataset)
    imgx, imgy = rw.get_RasterXY(tifDataset)
    tifArr = rw.get_RasterArr(tifDataset, imgx, imgy)
    data = tifArr.reshape(4, imgx * imgy)
    blue = data[0, :][data[3, :] != 0].astype(np.float32)
    green = data[1, :][data[3, :] != 0].astype(np.float32)
    red = data[2, :][data[3, :] != 0].astype(np.float32)
    nir = data[3, :][data[3, :] != 0].astype(np.float32)

    ndvi = (nir - red) / (nir + red)
    # ndvi = (data[3, :] - data[2, :]) / (data[3, :] + data[2, :])
    print(ndvi.shape, blue.shape)
    output_img1 = np.zeros(imgx * imgy, dtype='float32') + np.nan
    temp = np.vstack((blue, green, red, nir, ndvi)).T.astype(np.float64)
    print(temp.shape, nir.shape)
    result = model.predict(temp)
    output_img1[data[3, :] != 0] = result
    output_img1 = output_img1.reshape(tifArr.shape[1], tifArr.shape[2])
    result_time = time.time()
    print(result_time - start_time)
    rw.write_tif(output_img1, project, transform, outPath)


def GetMinArr(Dataset1, Dataset2, outPath):
    prosrs, transform1 = rw.get_GeoInformation(Dataset1)
    transform2 = rw.get_GeoInformation(Dataset2)[1]
    imgx, imgy = rw.get_RasterXY(Dataset1)
    arr1 = rw.get_RasterArr(Dataset1, imgx, imgy)
    imgx, imgy = rw.get_RasterXY(Dataset2)
    arr2 = rw.get_RasterArr(Dataset2, imgx, imgy)
    if len(arr1.shape) > 2:
        arr1 = arr1[0]
    if len(arr2.shape) > 2:
        arr2 = arr2[0]
    print(arr1.shape, arr2.shape)
    print(transform1)
    print(transform2)
    lt_lon1, lt_lat1 = rw.imagexy2geo(Dataset1, 0, 0)
    lt_lon2, lt_lat2 = rw.imagexy2geo(Dataset2, 0, 0)
    rb_lon1, rb_lat1 = rw.imagexy2geo(Dataset1, arr1.shape[0], arr1.shape[1])
    rb_lon2, rb_lat2 = rw.imagexy2geo(Dataset2, arr2.shape[0], arr2.shape[1])
    # rt_lon1, rt_lat1 = rw.imagexy2geo(Dataset1, 0, arr1.shape[1])
    # rt_lon2, rt_lat2 = rw.imagexy2geo(Dataset2, 0, arr2.shape[1])
    # lb_lon1, lb_lat1 = rw.imagexy2geo(Dataset1, arr1.shape[0], 0)
    # lb_lon2, lb_lat2 = rw.imagexy2geo(Dataset2, arr2.shape[0], 0)

    lon_list = []
    lat_list = []
    if lt_lon1 - lt_lon2 <= 0:

        lon_list.append(lt_lon2)
    else:
        lon_list.append(lt_lon1)

    # if lb_lon1 - lb_lon2 < 0:
    #     lon_list.append(lb_lon2)
    # else:
    #     lon_list.append(lb_lon1)
    #
    # if rt_lon1 - rt_lon2 < 0:
    #     lon_list.append(rt_lon2)
    # else:
    #     lon_list.append(rt_lon1)

    if rb_lon1 - rb_lon2 <= 0:
        lon_list.append(rb_lon2)
    else:
        lon_list.append(rb_lon1)
    # 维度变化
    if lt_lat1 - lt_lat2 <= 0:

        lat_list.append(lt_lat2)
    else:
        lat_list.append(lt_lat1)

    # if lb_lat1 - lb_lat2 < 0:
    #     lon_list.append(lb_lat2)
    # else:
    #     lon_list.append(lb_lat1)
    #
    # if rt_lat1 - rt_lat2 < 0:
    #     lon_list.append(rt_lat2)
    # else:
    #     lon_list.append(rt_lat1)

    if rb_lat1 - rb_lat2 <= 0:
        lat_list.append(rb_lat2)
    else:
        lat_list.append(rb_lat1)


    print('-' * 50)
    print(lon_list)
    print('-' * 50)
    print(lat_list)
    print('-' * 50)

    # 两种情况：
    # 第一种：一张影像的左上角在另一个影像的里面
    # a = 3
    # print((~(a > 5) & (a > 2)))
    # exit(0)

    _, geosrs = rw.defineSRS(4326)
    min_row, min_col = rw.geo2imagexy(lon_list[0], lat_list[0], transform1)
    max_row, max_col = rw.geo2imagexy(lon_list[1], lat_list[1], transform1)
    arr1 = rw.get_RasterArr(Dataset1, lt_x=int(min_row), lt_y=int(min_col), rt_x=int(max_row) - int(min_row), rt_y=int(max_col) - int(min_col))
    print(min_row, min_col, max_row, max_col)
    min_row, min_col = rw.geo2imagexy(lon_list[0], lat_list[0], transform2)
    max_row, max_col = rw.geo2imagexy(lon_list[1], lat_list[1], transform2)
    arr2 = rw.get_RasterArr(Dataset2, lt_x=int(min_row), lt_y=int(min_col), rt_x=int(max_row) - int(min_row), rt_y=int(max_col) - int(min_col))
    print(min_row, min_col, max_row, max_col)
    print(arr1.shape, arr2.shape)
    result = arr1 - arr2
    transform = (lon_list[0], transform1[1], transform1[2], lat_list[0], transform1[4], transform1[5])
    rw.write_tif(result, prosrs, transform, outPath)


if __name__ == '__main__':

    # inputPath_dict = {
    #     0: glob.glob(r'J:\000 化工厂\Classifier\build\*.tif'),
    #     1: glob.glob(r'J:\000 化工厂\Classifier\forest\*.tif'),
    #     # 3: glob.glob(r'J:\000 化工厂\Classifier\water\*.tif'),
    #     2: glob.glob(r'J:\000 化工厂\Classifier\soil\*.tif'),
    # }
    # train_x, test_x, train_y, test_y = readTrainData(inputPath_dict)
    # forest, et, xgb, light = train(train_x, test_x, train_y, test_y)

    # joblib.dump(forest, r'J:\000 化工厂\Mymodel\forest.m')
    # model = joblib.load(r'J:\000 化工厂\Mymodel\forest.m')
    # ModelPredict(model, r"J:\000 化工厂\精校正结果\20231031_clip_resample.tif", r'J:\000 化工厂\Mymodel\ImageResult\20231031_clip_resample_rf.tif')
    #
    # # joblib.dump(et, r'J:\000 化工厂\Mymodel\et.m')
    # model = joblib.load(r'J:\000 化工厂\Mymodel\et.m')
    # ModelPredict(model, r"J:\000 化工厂\精校正结果\20231031_clip_resample.tif", r'J:\000 化工厂\Mymodel\ImageResult\20231031_clip_resample_et.tif')
    #
    # # joblib.dump(et, r'J:\000 化工厂\Mymodel\xgb.m')
    # model = joblib.load(r'J:\000 化工厂\Mymodel\xgb.m')
    # ModelPredict(model, r"J:\000 化工厂\精校正结果\20231031_clip_resample.tif", r'J:\000 化工厂\Mymodel\ImageResult\20231031_clip_resample_xgb.tif')
    #
    # # joblib.dump(light, r'J:\000 化工厂\Mymodel\light.m')
    # model = joblib.load(r'J:\000 化工厂\Mymodel\light.m')
    # ModelPredict(model, r"J:\000 化工厂\精校正结果\20231031_clip_resample.tif", r'J:\000 化工厂\Mymodel\ImageResult\20231031_clip_resample_lgbm.tif')

    # Dataset1 = rw.get_tifDataset(r"J:\000 化工厂\Mymodel\ImageResult\20230128_clip_xgb.tif")
    # Dataset2 = rw.get_tifDataset(r"J:\000 化工厂\Mymodel\ImageResult\20231031_clip_resample_xgb.tif")
    # GetMinArr(Dataset1, Dataset2, r'J:\000 化工厂\Mymodel\ImageResult\cha.tif')
    Dataset = rw.get_tifDataset(r'J:\000 化工厂\Mymodel\ImageResult\cha.tif')
    imgx, imgy = rw.get_RasterXY(Dataset)
    projection, transform = rw.get_GeoInformation(Dataset)
    arr = rw.get_RasterArr(Dataset, imgx, imgy)
    arr[arr != -2] = 0
    arr[arr == -2] = 1
    result, area = find_and_compute_areas(arr[0], 0.64)


    rw.write_tif(result, projection, transform, r'J:\000 化工厂\Mymodel\ImageResult\shibie.tif')