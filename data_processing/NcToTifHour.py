# --coding:utf-8--
import datetime
import pytz
import netCDF4 as nc
import os
import pandas as pd
import numpy as np
import math
from osgeo import gdal, osr, ogr, gdalconst
import dateutil.parser
import re
import string
import math



# 通过经纬度计算仿射变换参数
def lonlatToGeo(lon, lat, prosrs, geosrs):

    ct = osr.CoordinateTransformation(geosrs, prosrs)
    print('-' * 50)
    coords = ct.TransformPoint(lon, lat)
    print(coords)

    return coords[:2]


def defineSRS(reference_num):

    prosrs = osr.SpatialReference()
    prosrs.ImportFromEPSG(reference_num)
    geosrs = prosrs.CloneGeogCS()

    return prosrs, geosrs


def get_TransForm(lon, lat, projection, geosrs):

    min_lon, max_lon = lon[0], lon[-1]
    max_lat, min_lat = lat[0], lat[-1]
    ymax = lon.shape[0]    # 一共有多少列
    xmax = lat.shape[0]    # 一共有多少行
    # print(min_lon, max_lon)
    # print(min_lat, max_lat)
    # print('列数和行数', ymax, xmax)
    pixel_points = np.array([np.array([0, 0]),
                             np.array([0, ymax]),
                             np.array([xmax, ymax]),
                             np.array([xmax, 0])
                             ])
    # print(pixel_points)

    geo_point1 = lonlatToGeo(float(min_lat), float(min_lon), projection, geosrs)
    geo_point2 = lonlatToGeo(float(min_lat), float(max_lon), projection, geosrs)
    geo_point3 = lonlatToGeo(float(max_lat), float(max_lon), projection, geosrs)
    geo_point4 = lonlatToGeo(float(max_lat), float(min_lon), projection, geosrs)

    geo_points = np.array([geo_point1, geo_point2, geo_point3, geo_point4])
    # print(geo_points)
    # print('ymin', ymin, '左上', geo_point1)
    # print('ymax', ymax, '右下', geo_point2)
    # print('xmin', xmin, '右上', geo_point3)
    # print('xmax', xmax, '左下', geo_point4)

    n = len(pixel_points)
    pixelX_square = 0.0
    pixelX_pixelY = 0.0
    pixelY_square = 0.0
    pixelX = 0.0
    pixelY = 0.0

    pixelX_geoX = 0.0
    pixelY_geoX = 0.0
    pixelX_geoY = 0.0
    pixelY_geoY = 0.0

    geoX = 0.0
    geoY = 0.0

    for i in range(0, n):
        pixelX_square += math.pow(pixel_points[i, 1], 2)
        pixelX_pixelY += pixel_points[i, 1] * pixel_points[i, 0]
        pixelY_square += math.pow(pixel_points[i, 0], 2)
        pixelX += pixel_points[i, 1]
        pixelY += pixel_points[i, 0]

        pixelX_geoX += pixel_points[i, 1] * geo_points[i, 1]
        pixelY_geoX += pixel_points[i, 0] * geo_points[i, 1]
        pixelX_geoY += pixel_points[i, 1] * geo_points[i, 0]
        pixelY_geoY += pixel_points[i, 0] * geo_points[i, 0]

        geoX += geo_points[i, 1]
        geoY += geo_points[i, 0]

    a = np.array([[pixelX_square, pixelX_pixelY, pixelX], [pixelX_pixelY, pixelY_square, pixelY], [pixelX, pixelY, n]])
    b1 = np.array([[pixelX_geoX], [pixelY_geoX], [geoX]])
    b2 = np.array([[pixelX_geoY], [pixelY_geoY], [geoY]])

    at = np.linalg.inv(a)
    result1 = at.dot(b1)
    result2 = at.dot(b2)
    # 这里WGS_84和正弦曲线投影前后三个是颠倒过来的, 2在前是正弦曲线投影，1在前是84投影
    return result1[2, 0] - (round(result1[0, 0], 2) / 2), round(result1[0, 0], 2), result1[1, 0],\
           result2[2, 0] + (round(result2[1, 0], 2) / 2), result2[0, 0], round(result2[1, 0], 2)


def write_tif(data_list, projection, transform, outputPath):

    if 'int8' in data_list.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data_list.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    row, col = data_list.shape[0], data_list.shape[1]
    driver = gdal.GetDriverByName('GTiff')

    ds = driver.Create(outputPath, col, row, 1, datatype)
    print(transform)
    ds.SetProjection(projection.ExportToWkt())
    ds.SetGeoTransform(transform)

    print(transform)
    ds.GetRasterBand(1).WriteArray(data_list)
    ds.FlushCache()
    del ds


# 根据下标计算平均值
# 获取数据在文件中的的位置
def math_mean(pd_date_data, flag, use_value, fillvalue):
    data_arg = pd_date_data[flag:flag].values.reshape(-1)
    temp_list = []
    # 开始读数据
    print('全部数据形状', use_value.shape)
    temp_arr = None
    for arg in data_arg:
        if len(use_value.shape) >= 4:
            temp_arr = use_value[arg, 0, :, :]
            print('居然有4个！！！！！！！！！！！！！！！！')
        else:
            temp_arr = use_value[arg, :, :]
        print("合并前每个小时的数据形状", use_value[arg, :, :].shape)

        temp_arr[temp_arr == fillvalue] = np.nan
        temp_list.append(temp_arr)
    # 循环结束得到三维矩阵的品均值
    temp_list = np.array(temp_list)
    # 这里是三维的
    print('合并后的数据', temp_list.shape)
    result = np.nanmean(temp_list, axis=0)
    # 这里是二维的说明计算成功
    print('计算平均值', result.shape)

    return result


# 将nc文件转换为tif文件
def get_nc_mean_tif(nc_path, loc_time_zone, value_name, outputpath):
    """
    :param nc_path: nc文件路径
    :param loc_time_zone: timezone某地地方时
    :param value_name: 气象数据name
    :param outputpath: 输出文件路径
    :return:
    """
    os.getcwd()
    nc_data = nc.Dataset(nc_path)
    # 获取到nc文件中读取的日期信息
    time_information = str(nc_data.variables['time']).split('\n')

    nc_start_date = time_information[2].lstrip()[6:]
    utc_time_list = nc.num2date(nc_data.variables['time'][:], nc_start_date, only_use_cftime_datetimes=False).data

    # 获取地方时
    loc_time_list = []
    for i in range(len(utc_time_list)):
        utc_time = datetime.datetime.strptime(str(utc_time_list[i]), '%Y-%m-%d %H:%M:%S')
        loc_time = utc_time.astimezone(loc_time_zone)
        loc_time_list.append(loc_time)

    # 这里之后读数据取平均需要用到
    pd_date_data = pd.DataFrame([i for i in range(len(loc_time_list))], index=loc_time_list, columns=['arg'])


    # 需要所有数据
    # 获取文件中其他的数据，一般有经纬度，和下载的气象数据
    lon = nc_data.variables['longitude'][:].data
    lat = nc_data.variables['latitude'][:].data
    use_value = nc_data.variables[value_name][:].data
    print('刚读出数据的格式', use_value.shape)

    # 获取气象数据的fill_value offset factor
    value_information = str(nc_data.variables[value_name]).split('\n')

    fill_value = None
    for get_value in value_information:
        if '_FillValue' in get_value:
            fill_value = int(get_value.split(': ')[1])

    # 获取最大最小经纬度(因为是矩形)，以及行列数
    prosrs, geosrs = defineSRS(4326)
    LonMin, LatMax, LonMax, LatMin = [np.min(lon), np.max(lat), np.max(lon), np.min(lat)]
    # 设置图像分辨率
    Lon_Res = (np.max(lon) - np.min(lon)) / (len(lon) - 1)
    Lat_Res = (np.max(lat) - np.min(lat)) / (len(lat) - 1)
    N_Lat = len(lat)
    N_Lon = len(lon)
    transform = (LonMin, Lon_Res, 0, LatMax, 0, -Lat_Res)
    transform = (LonMin - (Lon_Res * 0.5), Lon_Res, 0, LatMax + (Lat_Res * 0.5), 0, -Lat_Res)

    print(pd_date_data.index.values[0])

    # print(use_value)
    for i in range(use_value.shape[0]):
        print(os.path.join(outputpath, str(pd_date_data.index.values[i])[:-16] + '_' + value_name + '.tif'))
        write_tif(use_value[i], prosrs, transform, os.path.join(outputpath, str(pd_date_data.index.values[i])[:-16] + '_' + value_name + '.tif'))




def get_nc10_mean_tif(nc_path, loc_time_zone, value_name, ws_or_wd, outputpath):
    """

    :param nc_path: nc文件路径
    :param loc_time_zone: timezone某地地方时
    :param value_name: 气象数据name
    :param outputpath: 输出文件路径
    :return:
    """
    os.getcwd()
    ncu_data = nc.Dataset(nc_path)
    v10 = nc_path.replace('_10m_u_component_of_winddownload.netcdf1', '_10m_v_component_of_winddownload.netcdf1')
    v10 = v10.replace('10m_u', '10m_v')
    ncv_data = nc.Dataset(v10)
    # 获取到nc文件中读取的日期信息
    time_information = str(ncu_data.variables['time']).split('\n')

    nc_start_date = time_information[2].lstrip()[6:]
    utc_time_list = nc.num2date(ncu_data.variables['time'][:], nc_start_date, only_use_cftime_datetimes=False).data

    # 获取地方时
    loc_time_list = []
    for i in range(len(utc_time_list)):
        utc_time = datetime.datetime.strptime(str(utc_time_list[i]), '%Y-%m-%d %H:%M:%S')
        loc_time = utc_time.astimezone(loc_time_zone)
        loc_time_list.append(loc_time)


    # 这里之后读数据取平均需要用到
    pd_date_data = pd.DataFrame([i for i in range(len(loc_time_list))], index=loc_time_list, columns=['arg'])

    # 获取文件中其他的数据，一般有经纬度，和下载的气象数据
    lon = ncu_data.variables['longitude'][:].data
    lat = ncu_data.variables['latitude'][:].data
    useu_value = ncu_data.variables[value_name][:].data
    usev_value = ncv_data.variables['v10'][:].data

    print('刚读出数据的格式', useu_value.shape)

    # 获取气象数据的fill_value offset factor
    value_information = str(ncu_data.variables[value_name]).split('\n')
    fill_value = None
    for get_value in value_information:
        if '_FillValue' in get_value:
            fill_value = int(get_value.split(': ')[1])
    print(fill_value)
    useu_value[useu_value == fill_value] = np.nan
    usev_value[usev_value == fill_value] = np.nan
    # 获取最大最小经纬度(因为是矩形)，以及行列数
    prosrs, geosrs = defineSRS(4326)
    # 获取四个脚点坐标
    LonMin, LatMax, LonMax, LatMin = [np.min(lon), np.max(lat), np.max(lon), np.min(lat)]
    # 设置图像分辨率
    Lon_Res = (np.max(lon) - np.min(lon)) / (len(lon) - 1)
    Lat_Res = (np.max(lat) - np.min(lat)) / (len(lat) - 1)
    N_Lat = len(lat)
    N_Lon = len(lon)

    # 数据 geotransform
    transform = (LonMin - (Lon_Res * 0.5), Lon_Res, 0, LatMax + (Lat_Res * 0.5), 0, -Lat_Res)


    #计算风速或风向
    if ws_or_wd == 'ws':
        wind_data = usev_value * usev_value + useu_value * useu_value
        wind_data = np.sqrt(wind_data)
    else:
        aa = 180 + 180 / math.pi * np.arctan2(usev_value, useu_value)
        wind_data = np.mod(aa, 360)
    for i in range(useu_value.shape[0]):
        print(str(pd_date_data.index.values[0])[:13])
        print(value_name)
        print(wind_data[i].shape, '111')
        print(os.path.join(outputpath, str(pd_date_data.index.values[i])[:13] + '_' + ws_or_wd + '.tif'))
        write_tif(wind_data[i], prosrs, transform, os.path.join(outputpath, str(pd_date_data.index.values[i])[:13] + '_' + str(ws_or_wd) + '.tif'))


if __name__ == '__main__':
    search_dict = {'10_u': 'u10',
                   '10_v': 'v10',
                   'blh': 'blh',
                   'r': 'r',
                   'sp': 'sp',
                   't2m': 't2m',
                   'tp': 'tp'
                   }

    timezone = [pytz.timezone('Asia/Shanghai')]


    path = [r'D:\air\datalera5气象数据']

    output_dir = [r'J:\测试']

   

    city_flag = 0
    for city_path in path:
        citys = os.listdir(city_path)
        for value_dir in citys:
            if ('10_v' in value_dir):

                data_path = os.path.join(city_path, value_dir)
                for data in os.listdir(data_path):
                    suibian = os.path.join(data_path, data)

                    for true_data in os.listdir(suibian):
                        yongde = os.path.join(suibian, true_data)
                        aa = [v for k, v in search_dict.items() if k in data_path]
                        outputpath = os.path.join(output_dir[city_flag], 'wd')
                        print(os.path.join(suibian, true_data))
                        get_nc10_mean_tif(yongde, timezone[city_flag], aa[0], 'wd', outputpath)
