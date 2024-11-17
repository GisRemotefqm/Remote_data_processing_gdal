# --coding:utf-8--
import numpy as np
from osgeo import gdal
import os
from scipy.interpolate import Rbf
import glob
from DataMath import ReadWrite_h5
# 已知点的坐标和值

def BatchTiff2IDW(inpaths, outpath):

    paths = inpaths
    for path in paths:
        dataset = gdal.Open(path)

        arr = dataset.ReadAsArray()
        transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        arr[arr == -1] = np.nan
        # 非空值的位置
        non_nan_indices = np.where(~np.isnan(arr))
        non_nan_indices_values = arr[non_nan_indices]
        non_nan_indices_arg = np.argwhere(~np.isnan(arr))
        non_nan_indices_values = non_nan_indices_values[:, np.newaxis]
        # print(non_nan_indices_values.shape, non_nan_indices_arg.shape)
        known_points = np.hstack((non_nan_indices_arg, non_nan_indices_values))
        # print(known_points.shape)

        # 空值的位置
        nan_indices = np.where(np.isnan(arr))
        nan_indices_values = arr[nan_indices]
        nan_indices_arg = np.argwhere(np.isnan(arr))
        # print(nan_indices_values.shape, nan_indices_arg.shape)
        unknown_points = nan_indices_arg
        # print(arr)
        # print((arr == np.nan).all())

        try:
        # 计算反距离权重插值
            rbf = Rbf(known_points[:, 0], known_points[:, 1], known_points[:, 2], function='inverse')
            interpolated_values = rbf(unknown_points[:, 0], unknown_points[:, 1])
            arr[nan_indices] = interpolated_values
            ReadWrite_h5.write_tif(arr, projection, transform, os.path.join(outpath, os.path.basename(path)))
        except:
            ReadWrite_h5.write_tif(arr, projection, transform, os.path.join(outpath, os.path.basename(path)))



paths = glob.glob(r".\*.tif")
outpath = r'.\IDW'
