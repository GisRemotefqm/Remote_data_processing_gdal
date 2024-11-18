import numpy as np
from osgeo import gdal, gdalconst
import osgeo.osr as osr
import os
import csv
import xml.dom.minidom
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta


def get_file_list(file_path, file_type):
    file_path = file_path
    file_list = []
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(file_type + " files not find in {}".format(file_path))
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == '.' + file_type:
                file_list.append([file, os.path.join(root, file)])
    return file_list


def process_kj(in_path, name, out_path):
    csv_list = get_file_list(in_path, 'csv')
    df_list = []
    for cn, cp in csv_list:
        print(cn)
        csv_df = pd.read_csv(filepath_or_buffer=cp, skiprows=4)
        csv_df['site'] = name
        # csv_df = csv_df[['site', 'Date(dd/mm/yyyy)', 'Time(hh:mm:ss)', 'AOT_440', 'AOT_675', '440-675Angstrom']]
        csv_df = csv_df[['site', 'Time(hh:mm:ss)', 'AOT_440', 'AOT_675', '440-675Angstrom']]
        csv_df.dropna(subset=['AOT_440', 'AOT_675'], inplace=True)
        csv_df = csv_df[(csv_df['AOT_440'] > 0) & (csv_df['AOT_675'] > 0)]
        csv_df = csv_df.reset_index(drop=True)

        aod_440 = np.array(csv_df['AOT_440'])
        aod_675 = np.array(csv_df['AOT_675'])

        ae = -np.log(aod_440/aod_675)/np.log(440/675)
        csv_df['AE'] = ae

        aod_550 = aod_440 * np.power(550/440, -ae)
        csv_df['AOT_550'] = aod_550

        csv_df['AOT_550_KJ'] = 0

        for i in range(0, csv_df.shape[0]):
            if not np.isnan(csv_df.loc[i, '440-675Angstrom']):
                csv_df.loc[i, 'AOT_550_KJ'] = aod_440[i] * np.power(550/440, -csv_df.loc[i, '440-675Angstrom'])
                n = 1

        # csv_df['Date(dd/mm/yyyy)'] = pd.to_datetime(csv_df['Date(dd/mm/yyyy)'], format="%d/%m/%Y").dt.date
        csv_df['Date(dd/mm/yyyy)'] = datetime.strptime('20'+cn[0:6], '%Y%m%d').date()
        csv_df.rename(columns={'Date(dd/mm/yyyy)': 'Date', 'Time(hh:mm:ss)': 'Time'}, inplace=True)
        df_list.append(csv_df)

    out_df = pd.concat(df_list)
    out_df.to_csv(path_or_buf=out_path, index=False)


def create_tiff(img_path, im_data, im_geotrans, im_proj):
    # gdal数据类型
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(img_path, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


def get_keyword_mean(keyword, tiff_path, out_path, control_num):
    tiff_list = get_file_list(tiff_path, 'tif')

    pm25 = gdal.Open(tiff_list[0][1])
    geotrans = pm25.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    proj = pm25.GetProjection()  # 地图投影信息，字符串表示

    n_columns = pm25.RasterXSize
    n_rows = pm25.RasterYSize

    out_tiff = out_path + '\\' + keyword + '_mean.tiff'
    empty_array = np.zeros((n_rows, n_columns, 1), dtype=np.float32) + np.nan

    for t_n, t_p in tiff_list:
        dataset = gdal.Open(t_p)
        columns = dataset.RasterXSize
        rows = dataset.RasterYSize

        data = dataset.ReadAsArray(0, 0, columns, rows)
        data[data <= 0] = np.nan
        data_3d = data.reshape(data.shape[0], data.shape[1], 1)
        empty_array = np.concatenate((empty_array, data_3d), axis=2)

    mean_array = np.nanmean(empty_array, axis=2)
    mean_array[mean_array <= 0.01] = np.nan
    mean_array[mean_array >= 1.0] = np.nan

    idx = empty_array > 0.0
    valid_num = np.sum(idx, axis=2)
    valid_idx = (valid_num >= control_num)

    out_array = np.zeros((n_rows, n_columns), dtype=np.float32) + np.nan
    out_array[valid_idx] = mean_array[valid_idx]

    create_tiff(out_tiff, out_array, geotrans, proj)
    print(out_tiff)
    print('------------------------------------------------')


def get_color(in_path, out_path):
    """
    得到真彩色图像
    """

    tif_list = get_file_list(in_path, 'tiff')
    data_list = []
    for tn, tp in tif_list:
        dataset = gdal.Open(tp)
        t_columns = dataset.RasterXSize
        t_rows = dataset.RasterYSize
        data = dataset.ReadAsArray(0, 0, t_columns, t_rows)
        data_list.append(data)

        geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    color_data = np.array(data_list)

    create_tiff(out_path, color_data, geotrans, proj)


def defineSin():
    sin_WKT = 'PROJCS["World_Sinusoidal",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    esri_prosrs = osr.SpatialReference()
    esri_prosrs.ImportFromWkt(sin_WKT)
    # esri_prosrs.ImportFromWkt('PROJCS["World_Sinusoidal",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],UNIT["Meter",1.0]]')

    esri_geosrs = esri_prosrs.CloneGeogCS()
    return esri_prosrs, esri_geosrs


def sift():
    data_obj = gdal.Open(r"J:\GF5_DPC_20191202_008334_L10000028231_B490.h5")
    dataset_list = data_obj.GetSubDatasets()
    # aa = dataset_list[0][0]
    I490 = gdal.Open(dataset_list[0][0])
    img_x = I490.RasterXSize
    img_y = I490.RasterYSize

    pro, geo = defineSin()
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)  # 定义输出的坐标系为 WGS84
    # I490.SetGeoTransform((0, 1, 0, 0, 0, 1))
    I490.SetProjection('')

    print(I490.GetProjection())
    print(I490.GetGeoTransform())
    print(I490.GetMetadata())

    # I490_data = I490.GetRasterBand(1).ReadAsArray(0, 0, img_x, img_y)
    # I490_data = I490_data * 0.0001
    # I490_data[(I490_data == 3.2767) | (I490_data == -3.2767)] = 0

    lat_dataset = gdal.Open(dataset_list[8][0])
    lon_dataset = gdal.Open(dataset_list[9][0])
    lat_data = lat_dataset.ReadAsArray(0, 0, img_x, img_y)
    lon_data = lon_dataset.ReadAsArray(0, 0, img_x, img_y)

    lat = lat_data[(lat_data < 90) & (lat_data > -90)]
    lon = lon_data[(lon_data < 180) & (lon_data > -180)]

    lat_arg = np.argwhere((lat_data < 90) & (lat_data > -90))

    # 将 原始经纬度 转换为 正弦投影经纬度

    ct = osr.CoordinateTransformation(geo, pro)
    lat2 = np.zeros((lat.shape[0]))
    lon2 = np.zeros((lon.shape[0]))

    for i in tqdm(range(0, lat.shape[0])):
    # for i in range(0, lat.shape[0]):
        coords = ct.TransformPoint(float(lat[i]), float(lon[i]))
        lat2[i] = coords[0]
        lon2[i] = coords[1]

    # 添加控制点
    gcp_list = []
    for j in tqdm(range(0, int(lat.shape[0] / 50000))):
    # for j in range(0, lat.shape[0]):
        # print((lon2[0] - lon2[-1]) / (float(lat_arg[0, 1]) - float(lat_arg[-1, 0])))
        gcp = gdal.GCP(float(lat2[j]), float(lon2[j]), 0, float(lat_arg[j, 1]), float(lat_arg[j, 0]))
        gcp_list.append(gcp)

    # I490.SetProjection(pro.ExportToWkt())

    tiff_driver = gdal.GetDriverByName('MEM')
    temp_ds = tiff_driver.CreateCopy('', I490, 0)
    print(gcp_list[0])
    temp_ds.SetGCPs(gcp_list, pro)


    gdal.Warp(r"J:\CM1_DPC1_20230609_B490.tif",
              temp_ds, format='GTiff', tps=True, dstSRS=out_srs,
              resampleAlg=gdalconst.GRA_Bilinear,
              dstNodata=0, srcNodata=-32767)
    n = 1


if __name__ == '__main__':
    a = np.array([[0, 0, 0],
                  [10, 10, 10],
                  [20, 20, 20],
                  [30, 30, 30]])
    b = np.array([1, 2, 3])
    print(a.shape)
    print(b.shape)
    print((a * b).shape)
    arr1 = np.arange(1,101)
    arr1 = arr1.reshape(25, 4)

    arr2 = np.arange(1, 201)
    arr2 = arr2.reshape(5, 10, 4)
    print(arr2)
    arr2 = arr2.transpose(2, 1, 0)
    print(arr2)
    arr2 = arr2.reshape(4, 50)
    arr3 = np.multiply(arr1.T[:, :, np.newaxis], arr2[:, np.newaxis, :])
    print(arr3.shape)
    exit(0)
    sift()

    # cm1_df = pd.read_csv(r'E:\CM1\Site_AOD.csv')
    # # cm1_df['AERONET'] = 0
    # cm1_df['KongJi'] = 0
    #
    # df_list = []
    # # csv_list = get_file_list(r'E:\CM1\231119 Aeronet\3d', 'csv')
    # csv_list = get_file_list(r'E:\CM1\231118 DPC\CE318\step1', 'csv')
    # for cn, cp in csv_list:
    #     csv_df = pd.read_csv(cp)
    #     df_list.append(csv_df)
    #
    # aeronet_df = pd.concat(df_list)
    # aeronet_df = aeronet_df.reset_index(drop=True)
    #
    # for i in range(0, cm1_df.shape[0]):
    #     site_name = cm1_df.loc[i, 'site']
    #     site_date = cm1_df.loc[i, 'start_date']
    #
    #     start_time = datetime.strptime(cm1_df.loc[i, 'start_time'], '%H:%M:%S')
    #     end_time = datetime.strptime(cm1_df.loc[i, 'end_time'], '%H:%M:%S')
    #     start_2 = (start_time - timedelta(hours=2)).time()
    #     end_2 = (end_time + timedelta(hours=2)).time()
    #
    #     day_df = aeronet_df[(aeronet_df['site'] == site_name) & (aeronet_df['Date'] == site_date)]
    #     day_df = day_df.reset_index(drop=True)
    #
    #     if day_df.shape[0] > 0:
    #         time_df = pd.to_datetime(day_df['Time']).dt.time
    #         select_df = day_df[(time_df > start_2) & (time_df < end_2)]
    #         if select_df.shape[0] > 0:
    #             # aod = np.array(select_df['AOD_T_550'])
    #             aod = np.array(select_df['AOT_550'])
    #             aod_mean = np.mean(aod)
    #             # cm1_df.loc[i, 'AERONET'] = aod_mean
    #             cm1_df.loc[i, 'KongJi'] = aod_mean
    #
    # cm1_df.to_csv(r'E:\CM1\Site_AOD_KongJi.csv')

    # process_kj(r'E:\CM1\231118 DPC\CE318\LuanCheng', 'LuanCheng', r'E:\CM1\231118 DPC\CE318\LuanCheng.csv')

    # get_keyword_mean('670',
    #                  r'E:\gongxing1(diyipi)\CM1_DPC1_20230529_004520_L10000416883\670',
    #                  r'E:\gongxing1(diyipi)\CM1_DPC1_20230529_004520_L10000416883\angle_mean',
    #                  3)

    # get_color(r'E:\gongxing1(diyipi)\CM1_DPC1_20230529_004520_L10000416883\angle_mean',
    #           r'E:\gongxing1(diyipi)\CM1_DPC1_20230529_004520_L10000416883\angle_mean\490-565-670.tiff')

