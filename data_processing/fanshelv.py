import glob
import os
import numpy as np
import pandas as pd
from osgeo import gdal, gdalconst
from DataMath import ReadWrite_h5
import xml.dom.minidom
import math

# 2020 GF-1 WFV gain
WFV1_2020 = [0.19319, 0.16041, 0.12796, 0.13405]
WFV2_2020 = [0.2057, 0.1648, 0.126, 0.1187]
WFV3_2020 = [0.2106, 0.1825, 0.1346, 0.1187]
WFV4_2020 = [0.2522, 0.2029, 0.1528, 0.1031]


def ReadXmlToTransform(inputpath, projection, geosrs):
    dom = xml.dom.minidom.parse(inputpath)

    root = dom.documentElement

    start_obj = root.getElementsByTagName('TopLeftLatitude')[0]
    TopLeftLatitude = start_obj.firstChild.data
    start_obj2 = root.getElementsByTagName('TopLeftLongitude')[0]
    TopLeftLongitude = start_obj2.firstChild.data
    start_obj3 = root.getElementsByTagName('TopRightLatitude')[0]
    TopRightLatitud = start_obj3.firstChild.data
    start_obj4 = root.getElementsByTagName('TopRightLongitude')[0]
    TopRightLongitude = start_obj4.firstChild.data
    start_obj5 = root.getElementsByTagName('BottomRightLatitude')[0]
    BottomRightLatitude = start_obj5.firstChild.data
    start_obj6 = root.getElementsByTagName('BottomRightLongitude')[0]
    BottomRightLongitude = start_obj6.firstChild.data
    start_obj7 = root.getElementsByTagName('BottomLeftLatitude')[0]
    BottomLeftLatitude = start_obj7.firstChild.data
    start_obj8 = root.getElementsByTagName('BottomLeftLongitude')[0]
    BottomLeftLongitude = start_obj8.firstChild.data

    start_obj9 = root.getElementsByTagName('WidthInPixels')[0]
    WidthInPixels = start_obj9.firstChild.data
    start_obj10 = root.getElementsByTagName('HeightInPixels')[0]
    HeightInPixels = start_obj10.firstChild.data

    pixel_points = np.array([np.array([0, 0]), np.array([0, float(WidthInPixels)]),
                             np.array([float(HeightInPixels), float(WidthInPixels)]), np.array([float(HeightInPixels), 0])])

    geo_point1 = ReadWrite_h5.lonlatToGeo(float(TopLeftLatitude), float(TopLeftLongitude), projection, geosrs)
    geo_point2 = ReadWrite_h5.lonlatToGeo(float(TopRightLatitud), float(TopRightLongitude), projection, geosrs)
    geo_point3 = ReadWrite_h5.lonlatToGeo(float(BottomRightLatitude), float(BottomRightLongitude), projection, geosrs)
    geo_point4 = ReadWrite_h5.lonlatToGeo(float(BottomLeftLatitude), float(BottomLeftLongitude), projection, geosrs)

    geo_points = np.array([geo_point1, geo_point2, geo_point3, geo_point4])

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
    return (result1[2, 0], result1[0, 0], result1[1, 0], result2[2, 0], result2[0, 0], result2[1, 0]), WidthInPixels, HeightInPixels


def math_rc(data_arr, gain):

    result = data_arr * gain

    return result


def write_tif(data_list, projection, transform, outputpath):

    if 'int8' in data_list.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data_list.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    row, col = data_list.shape[1], data_list.shape[2]

    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(outputpath, col, row, data_list.shape[0], datatype)
    output_ds.SetGeoTransform(transform)
    output_ds.SetProjection(projection.ExportToWkt())
    for i in range(data_list.shape[0]):

        output_ds.GetRasterBand(i + 1).WriteArray(data_list[i])
        output_ds.FlushCache()
    del output_ds
    print(outputpath, '计算完成')


def readtxt(txtpath, transform, projection, geosrs, imgx, imgy):
    txt_data = pd.read_table(txtpath, sep=' ', encoding='utf-8')
    lat = txt_data['lat'].to_numpy(float)
    lon = txt_data['lon'].to_numpy(float)
    for i in range(lat.shape[0]):

        x, y = ReadWrite_h5.geo2imagexy(lon[i], lat[i], transform)
        if ((x <= imgx) & (x >= 0)) & ((y <= imgy) & (y >= 0)):
            print('说明存在点在影像中')
            return True

    else:
        print('说明不存在点在影像中')
        return False


def sdss(ss):
    resamp_path = r'.\image'
    lujing = os.path.split(ss)[0]
    start, end = os.path.splitext(os.path.basename(ss))
    date = os.path.split(lujing)[1]
    # temp_path = os.path.join(resamp_path, date)

    restart = start
    if restart < '2016-08':
        print(restart)
        return
    repath = os.path.join(resamp_path, restart + '_resample.tif')

    d = ReadWrite_h5.get_tifDataset(ss)
    print(d)
    if d is None:
        print('说明文件有错误')
        return
    pr, tr = ReadWrite_h5.get_GeoInformation(d)
    gdal.Warp(repath, ss,
                    dstSRS=pr,
                    resampleAlg=gdalconst.GRA_NearestNeighbour,
                    # dstNodata=0,
                    # srcNodata=0,
                    # xRes=0.0089831528,
                    # yRes=0.0089831528,
                    xRes=0.001,
                    yRes=0.001,
              )
    print(repath, '重采样完成')
    del d




if __name__ == '__main__':

    output = r'.\result'

    for data in os.listdir(output):
        if os.path.splitext(data)[1] != '.tif':
            continue

        path = os.path.join(output, data)

        sdss(path)





