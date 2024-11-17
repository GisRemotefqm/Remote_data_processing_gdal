import glob
import numpy as np
import osgeo.gdal as gdal
import osgeo.osr as osr
import os
import math
import re


def imagexy2geo(dataset, row, col):

    '''
        根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
        :param dataset: GDAL地理数据
        :param row: 像素的行号
        :param col: 像素的列号
        :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def get_h5filelist(inputpath):
    try:
        data_obj = gdal.Open(inputpath)
    except:
        inputpath = str(input('重新输入路径'))
        data_obj = gdal.Open(inputpath, gdal.GA_Update)

    datapath_list = data_obj.GetSubDatasets()

    return datapath_list


def get_h5Dataset(datapath_list, readname):
    for datapath in datapath_list:
        name = os.path.split(datapath[0])[-1]
        if readname == name:
            dataset = gdal.Open(datapath[0], gdal.GA_Update)
            print(type(dataset))
            # print(dataset.GetMetadata())

            return dataset


def get_tifDataset(inputpath):
    dataset = gdal.Open(inputpath)

    return dataset


def get_GeoInformation(dataset):
    projection = dataset.GetProjection()
    transform = dataset.GetGeoTransform()

    return projection, transform


def get_RasterXY(dataset):
    img_X = dataset.RasterXSize
    img_Y = dataset.RasterYSize

    return img_X, img_Y


def get_RasterArr(dataset, rt_x, rt_y, lt_x=0, lt_y=0):
    """
    读取影像
    :param dataset:数据
    :param rt_x: 底部x
    :param rt_y: 底部y
    :param lt_x: 顶部x
    :param lt_y: 顶部y
    :return: 矩阵
    """
    data_list = []
    band = dataset.RasterCount
    for i in range(band):
        data_list.append(dataset.GetRasterBand(i + 1).ReadAsArray(lt_x, lt_y, rt_x, rt_y))

    return np.array(data_list)


def lonlatToGeo(lon, lat, prosrs, geosrs):
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    print('-' * 50)
    coords = ct.TransformPoint(lon, lat)
    print(coords)

    return coords[:2]


def geo2imagexy(lon, lat, transform):
    if (transform[0] > 180):
        re_transform = (transform[0] - 360, transform[1], transform[2],
                        transform[3], transform[4], transform[5])

        a = np.array([[re_transform[1], re_transform[2]], [re_transform[4], re_transform[5]]])
        b = np.array([lon - re_transform[0], lat - re_transform[3]])
    elif (transform[0] < -180):
        re_transform = (transform[0] + 360, transform[1], transform[2],
                        transform[3], transform[4], transform[5])
        a = np.array([[re_transform[1], re_transform[2]], [re_transform[4], re_transform[5]]])
        b = np.array([lon - re_transform[0], lat - re_transform[3]])
    else:
        a = np.array([[transform[1], transform[2]], [transform[4], transform[5]]])
        b = np.array([lon - transform[0], lat - transform[3]])

    return np.linalg.solve(a, b)


def defineSin():
    sin_WKT = 'PROJCS["World_Sinusoidal",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    esri_prosrs = osr.SpatialReference()
    esri_prosrs.ImportFromWkt(sin_WKT)

    esri_prosrs.ImportFromWkt(
        'PROJCS["World_Sinusoidal",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],UNIT["Meter",1.0]]')
    esri_geosrs = esri_prosrs.CloneGeogCS()
    return esri_prosrs, esri_geosrs


def defineSRS(reference_num):
    prosrs = osr.SpatialReference()
    prosrs.ImportFromEPSG(reference_num)
    geosrs = prosrs.CloneGeogCS()

    return prosrs, geosrs


def get_TransForm(lon_arr, lat_arr, projection, geosrs):
    lon_arg = np.argwhere((lon_arr > -180) & (lon_arr < 180) & (lon_arr != 0))

    ymin = (lon_arg[lon_arg[:, 1] == np.min(lon_arg[:, 1])])[0]
    xmin = lon_arg[lon_arg[:, 0] == np.min(lon_arg[:, 0])][-1]
    xmax = lon_arg[lon_arg[:, 0] == np.max(lon_arg[:, 0])][0]
    ymax = lon_arg[lon_arg[:, 1] == np.max(lon_arg[:, 1])][-1]
    pixel_points = np.array([ymin, xmin, ymax, xmax])
    print(lat_arr[ymin[0], ymin[1]])

    geo_point1 = lonlatToGeo(float(lat_arr[ymin[0], ymin[1]]), float(lon_arr[ymin[0], ymin[1]]), projection, geosrs)
    geo_point2 = lonlatToGeo(float(lat_arr[xmin[0], xmin[1]]), float(lon_arr[xmin[0], xmin[1]]), projection, geosrs)
    geo_point3 = lonlatToGeo(float(lat_arr[ymax[0], ymax[1]]), float(lon_arr[ymax[0], ymax[1]]), projection, geosrs)
    geo_point4 = lonlatToGeo(float(lat_arr[xmax[0], xmax[1]]), float(lon_arr[xmax[0], xmax[1]]), projection, geosrs)

    geo_points = np.array([geo_point1, geo_point2, geo_point3, geo_point4])
    print('ymin', ymin, '左上', geo_point1)
    print('ymax', ymax, '右下', geo_point2)
    print('xmin', xmin, '右上', geo_point3)
    print('xmax', xmax, '左下', geo_point4)

    n = len(pixel_points)
    print(pixel_points.shape, geo_points.shape)
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
    return result2[2, 0] - (result2[0, 0] / 2), result2[0, 0], result2[1, 0], \
           result1[2, 0] + (result1[1, 0] / 2), result1[0, 0], result1[1, 0]
    # if result2[2, 0] > result1[2, 0]:
    #     # 0, 1, 2, 3, 4, 5; x方向分辨率，旋转，地图左上角坐标， y方向分辨率， 旋转，地图左上角y坐标
    #     # print(round(result1[2, 0], 3), round(result1[0, 0], 2), round(result1[1, 0], 3), round(result2[2, 0], 3), round(result2[0, 0], 3), round(result2[1, 0], 2))
    #     return round(result2[2, 0], 2), round(result2[0, 0], 2), round(result2[1, 0], 2), round(result1[2, 0], 2), round(result1[0, 0], 2), round(result1[1, 0], 2)
    # else:
    #     return round(result1[2, 0], 2), round(result1[0, 0], 2), round(result1[1, 0], 2), round(result2[2, 0], 2), round(result2[0, 0], 2), round(result2[1, 0], 2)


# all arr
def write_tif(data_list, projection, transform, outputPath):
    if 'int8' in data_list.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data_list.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data_list.shape) > 2:

        row, col = data_list.shape[1], data_list.shape[2]

        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(outputPath, col, row, data_list.shape[0], datatype)

        ds.SetGeoTransform(transform)
        ds.SetProjection(projection)
        for i in range(data_list.shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(data_list[i])
        ds.FlushCache()
        del ds
    else:
        row, col = data_list.shape[0], data_list.shape[1]
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(outputPath, col, row, 1, datatype)
        ds.SetProjection(projection)
        ds.SetGeoTransform(transform)

        ds.GetRasterBand(1).WriteArray(data_list)
        ds.FlushCache()
        del ds


def writ_tif2(data_list, projection, transform, ouputpath):
    if 'int8' in data_list.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data_list.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    row, col = data_list.shape[1], data_list.shape[2]

    for i in range(data_list.shape[0]):
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(ouputpath + str(i) + '.tif', col, row, 1, datatype)

        ds.SetGeoTransform(transform)
        ds.SetProjection(projection.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(data_list[i])
        ds.FlushCache()
        del ds


def resampling(dataset, outputpath, projection, flag=1, nodata=0, new_nodata=-1):
    resamp = gdal.GRA_NearestNeighbour
    if flag == 1:
        resamp = gdal.GRA_NearestNeighbour
    elif flag == 2:
        resamp = gdal.GRA_Bilinear
    elif flag == 3:
        resamp = gdal.GRA_Cubic

    gdal.Warp(outputpath, dataset, dstSRS=projection,
              resampleAlg=resamp, srcNodata=nodata, dstNodata=new_nodata,
              xRes=0.001,
              yRes=0.001)


def batch_totif(inputpath, outputpath):
    pass


if __name__ == '__main__':

    for tifPath in glob.glob(r'.\*.tiff'):
        dataset = get_tifDataset(tifPath)
        print(dataset)
        pro, trans = get_GeoInformation(dataset)
        resampling(dataset,
                   os.path.join(r'.\sentinel_resample',
                                os.path.basename(tifPath)[:-4] + '_resample.tif'),
                   pro, flag=3)
