# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:38:02 2023

@author: JZY9264
"""

import numpy as np
import netCDF4 as nc
from osgeo import gdal, osr, ogr
import os
import glob
import datetime
from DataMath import ReadWrite_h5
# os.environ['PROJ_LIB'] = r'F:\anaconda3\envs\gdal\Library\share\proj'
path = r'H:\USA\cloud_data'

def write_tif(data_list, projection, transform, outputpath):

    if 'int8' in data_list.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data_list.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    row, col = data_list.shape[1], data_list.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(outputpath, col, row, 1, datatype)
    output_ds.SetGeoTransform(transform)
    output_ds.SetProjection(projection.ExportToWkt())
    output_ds.GetRasterBand(1).WriteArray(data_list[:, :].T)
    output_ds.FlushCache()
    del output_ds
    print(outputpath, '计算完成')


for data_name in os.listdir(path):

    if os.path.splitext(data_name)[1] != '.nc':
        continue
    data = os.path.join(path, data_name)

    f = nc.Dataset(data)

    var_lon = f['longitude'][:]
    #var_lat = f['latitude'][:][::-1]
    var_lat = f['latitude'][:]
    CTH = f['Cloud_Top_Height/Mean'][:]
    CTP = f['Cloud_Top_Pressure/Mean'][:]
    CTT = f['Cloud_Top_Temperature/Mean'][:]

    print(var_lat.shape)
    print(var_lon.shape)


    CTH_data_arr = np.asarray(CTH)
    CTP_data_arr = np.asarray(CTP)
    CTT_data_arr = np.asarray(CTT)
    print(CTH_data_arr.shape)
    print(CTH_data_arr, "000111110000")
    print(CTP_data_arr, "0000000")
    print(CTT_data_arr)


    # 影像的左上角和右下角坐标
    LonMin, LatMax, LonMax, LatMin = [var_lon.min(), var_lat.max(), var_lon.max(), var_lat.min()]
    # 分辨率计算
    N_Lat = var_lat.shape[0]
    N_Lon = var_lon.shape[0]
    Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)

    # 创建.tif文件
    im_bands = 1
    driver = gdal.GetDriverByName('GTiff')
    out_CTH_name = os.path.join(r"D:\测试文件", os.path.basename(data)[:-3] + 'CTH.tif')
    out_CTP_name = os.path.join(r"D:\测试文件", os.path.basename(data)[:-3] + 'CTP.tif')
    out_CTT_name = os.path.join(r"D:\测试文件", os.path.basename(data)[:-3] + 'CTT.tif')

    # out_tif = driver.Create(out_tif_name, N_Lon, N_Lat, im_bands, gdal.GDT_Float32)  # 创建框架
    # print(out_tif)
    print(N_Lat, N_Lon)
    # 设置影像的显示范围
    # Lat_Res一定要是-的
    geotransform = (LonMin - (Lon_Res / 2), Lon_Res, 0, LatMax + (Lat_Res / 2), 0, -Lat_Res)
    prosrs, geosrs = ReadWrite_h5.defineSRS(4326)
    print(geotransform)

    write_tif(CTH_data_arr, prosrs, geotransform, out_CTH_name)
    write_tif(CTP_data_arr, prosrs, geotransform, out_CTP_name)
    write_tif(CTT_data_arr, prosrs, geotransform, out_CTT_name)




# 数据写出

 
 
 
 
 
 
 
 
 