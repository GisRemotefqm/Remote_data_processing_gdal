from osgeo import gdal
import os
import glob
import numpy as np
import pandas as pd
from DataMath import ReadWrite_h5
import pysolar.solar as sun
import datetime
import xml.dom.minidom
import re
# GF1_WFV1_E0_2020 = [1968.63, 1849.19, 1571.46, 1079]
# GF1_WFV2_E0_2020 = [1955.11, 1847.22, 1569.45, 1087.87]
# GF1_WFV3_E0_2020 = [1956.62, 1840.46, 1541.45, 1084.06]
# GF1_WFV4_E0_2020 = [1968.08, 1841.53, 1540.8, 1069.6]

GF_1_WFV1_IR = [1968.7, 1859.7, 1560.1, 1078.1]
GF_1_WFV2_IR = [1957.3, 1857.6, 1560.1, 1079.3]
GF_1_WFV3_IR = [1960.1, 1854.2, 1557.1, 1080.7]
GF_1_WFV4_IR = [1969.2, 1855.7, 1557.7, 1078]

GF_6_WFV_2020_IR = [1952.16, 1847.16, 1554.76, 1074.06, 1412, 1267.39, 1762.64, 1736.92]
GF_6_WFV_2021_IR = [1952.16, 1847.43, 1554.76, 1074.06, 1412.0, 1267.39, 1792.64, 1736.92]
GF_6_WFV_2022_IR = [1952.16, 1847.43, 1554.76, 1074.06, 1412.0, 1267.39, 1792.64, 1736.92]
GF_6_WFV_2023_IR = [1952.16, 1847.43, 1554.76, 1074.06, 1412.0, 1267.39, 1792.64, 1736.92]

GF1_WFV1_2020_gain = [0.19319, 0.16041, 0.12796, 0.13405]
GF1_WFV2_2020_gain = [0.2057, 0.1648, 0.126, 0.1187]
GF1_WFV3_2020_gain = [0.2106, 0.1825, 0.1346, 0.1187]
GF1_WFV4_2020_gain = [0.2522, 0.2029, 0.1528, 0.1031]

GF1_WFV1_2021_gain = [0.1722, 0.1496, 0.1227, 0.1262]
GF1_WFV2_2021_gain = [0.1792, 0.1534, 0.1232, 0.1291]
GF1_WFV3_2021_gain = [0.2044, 0.1844, 0.1429, 0.1453]
GF1_WFV4_2021_gain = [0.2102, 0.1808, 0.1442, 0.1362]

GF1_WFV1_2022_gain = [0.1889, 0.1495, 0.1228, 0.1352]
GF1_WFV2_2022_gain = [0.1585, 0.1223, 0.0997, 0.1111]
GF1_WFV3_2022_gain = [0.2085, 0.1743, 0.1385, 0.1438]
GF1_WFV4_2022_gain = [0.2148, 0.1607, 0.1317, 0.1288]

GF1_WFV1_2023_gain = [0.1889, 0.1495, 0.1228, 0.1352]
GF1_WFV2_2023_gain = [0.1585, 0.1223, 0.0997, 0.1111]
GF1_WFV3_2023_gain = [0.2085, 0.1743, 0.1385, 0.1438]
GF1_WFV4_2023_gain = [0.2148, 0.1607, 0.1317, 0.1288]



def math_sza(xmlPath, tif_lon, tif_lat):

    dom = xml.dom.minidom.parse(xmlPath)
    root = dom.documentElement
    time = root.getElementsByTagName('ReceiveTime')[0].firstChild.data

    ymd, hms = time.split(' ')
    print(ymd, hms)
    year, month, day = ymd.split('-')
    hour, min, sec = hms.split(':')
    # t0 = datetime.datetime(int(tif_date[0:4]), int(tif_date[4:6]), int(tif_date[6:8]), hour, minute, 0, 0, tzinfo=datetime.timezone.utc)
    try:
        SZA_arr = 90 - sun.get_altitude(tif_lat, tif_lon,
                                        datetime.datetime(int(year), int(month), int(day), int(hour), int(min),
                                                          int(sec.split('.')[0]), int(sec.split('.')[1]),
                                                          tzinfo=datetime.timezone.utc))
    except:
        SZA_arr = 90 - sun.get_altitude(tif_lat, tif_lon,
                                        datetime.datetime(int(year), int(month), int(day), int(hour), int(min),
                                                          int(sec), tzinfo=datetime.timezone.utc))
    print(SZA_arr[int(SZA_arr.shape[0] / 2), int(SZA_arr.shape[1] / 2)])

    return SZA_arr


def math_TOA(xmlPath, tifPath, outputPath):

    tifDataset = ReadWrite_h5.get_tifDataset(tifPath)
    imgx, imgy = ReadWrite_h5.get_RasterXY(tifDataset)

    projection, transform = ReadWrite_h5.get_GeoInformation(tifDataset)
    tifArr = ReadWrite_h5.get_RasterArr(tifDataset, imgx, imgy)

    colSize = [np.arange(imgx) for n in range(imgy)]
    colSize = np.array(colSize) + 0.5
    colSize = colSize.reshape(imgy, imgx)

    rowSize = [np.arange(imgy) for n in range(imgx)]
    rowSize = np.array(rowSize)
    rowSize = rowSize.reshape(imgx, imgy)
    rowSize = rowSize.T + 0.5

    lon_arr, lat_arr = ReadWrite_h5.imagexy2geo(tifDataset, rowSize, colSize)

    xml = xmlPath
    SZA = math_sza(xml, lon_arr, lat_arr)
    SZA = np.deg2rad(SZA)

    zenith = np.deg2rad(SZA)
    # zenith = np.array([zenith, zenith, zenith, zenith])

    date = datetime.datetime.strptime(os.path.basename(tifPath).split('_')[4], '%Y%m%d')
    temp = date.timetuple()
    d = 1 + 0.0167 * np.sin((2 * np.pi * (temp.tm_yday - 93.5)) / 365)

    senor = os.path.basename(tifPath).split('_')[1]
    year = os.path.basename(tifPath).split('_')[4][:4]
    if senor == 'WFV1':
        ESUA = np.array(GF_1_WFV1_IR)[:, np.newaxis, np.newaxis]
        print(1)
    if senor == 'WFV2':
        ESUA = np.array(GF_1_WFV2_IR)[:, np.newaxis, np.newaxis]
    if senor == 'WFV3':
        ESUA = np.array(GF_1_WFV3_IR)[:, np.newaxis, np.newaxis]
    if senor == 'WFV4':
        ESUA = np.array(GF_1_WFV4_IR)[:, np.newaxis, np.newaxis]
    print(ESUA * np.sin(zenith))
    p = (np.pi * tifArr * np.power(d, 2)) / (ESUA * np.cos(zenith))

    ReadWrite_h5.write_tif(p, projection, transform, outputPath)


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
    output_ds.SetProjection(projection)
    for i in range(data_list.shape[0]):

        output_ds.GetRasterBand(i + 1).WriteArray(data_list[i])
        output_ds.FlushCache()
    del output_ds
    print(outputpath, '计算完成')


if __name__ == '__main__':

    tifPaths = glob.glob(r'K:\JDMD测试影像\RC\*.tif')
    outputDir = r'K:\JDMD测试影像\TOA'
    for tifPath in tifPaths:

        # tifDataset = ReadWrite_h5.get_tifDataset(tifPath)
        #
        # imgx, imgy = ReadWrite_h5.get_RasterXY(tifDataset)
        # projection, transform = ReadWrite_h5.get_GeoInformation(tifDataset)
        # tifArr = ReadWrite_h5.get_RasterArr(tifDataset, imgx, imgy)
        # print(os.path.basename(tifPath).split('_')[1])
        # senor= os.path.basename(tifPath).split('_')[1]
        # date = os.path.basename(tifPath).split('_')[4][:4]
        # if '2023' == date:
        #     if senor == 'WFV1':
        #         result = math_rc(tifArr, np.array(GF1_WFV1_2023_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV2':
        #         result = math_rc(tifArr, np.array(GF1_WFV2_2023_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV3':
        #         result = math_rc(tifArr, np.array(GF1_WFV3_2023_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV4':
        #         result = math_rc(tifArr, np.array(GF1_WFV4_2023_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        # elif '2022' == date:
        #     if senor == 'WFV1':
        #         result = math_rc(tifArr, np.array(GF1_WFV1_2022_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV2':
        #         result = math_rc(tifArr, np.array(GF1_WFV2_2022_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV3':
        #         result = math_rc(tifArr, np.array(GF1_WFV3_2022_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV4':
        #         result = math_rc(tifArr, np.array(GF1_WFV4_2022_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        # elif '2021' == date:
        #     if senor == 'WFV1':
        #         result = math_rc(tifArr, np.array(GF1_WFV1_2021_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV2':
        #         result = math_rc(tifArr, np.array(GF1_WFV2_2021_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV3':
        #         result = math_rc(tifArr, np.array(GF1_WFV3_2021_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV4':
        #         result = math_rc(tifArr, np.array(GF1_WFV4_2021_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        # elif '2020' == date:
        #     if senor == 'WFV1':
        #         result = math_rc(tifArr, np.array(GF1_WFV1_2020_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV2':
        #         result = math_rc(tifArr, np.array(GF1_WFV2_2020_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV3':
        #         result = math_rc(tifArr, np.array(GF1_WFV3_2020_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        #     elif senor == 'WFV4':
        #         result = math_rc(tifArr, np.array(GF1_WFV4_2020_gain)[:, np.newaxis, np.newaxis])
        #         write_tif(result, projection, transform, os.path.join(outputDir, os.path.basename(tifPath)[:-17] + '_ar.tif'))
        # else:
        #     print('可能是有错')
        print(tifPath)
        xmlDir = r'K:\JDMD测试影像'
        match = re.search('(GF1|GF6)?_(WFV\d|WFV)?(_(\d+))?_E(\d+\.\d+)_N(\d+\.\d+)_\d{8}_L1A\d{10}', os.path.basename(tifPath))
        print(os.path.join(xmlDir, match[0] + '.xml'))
        xmlPath = os.path.join(xmlDir, match[0] + '.xml')
        outputPath = os.path.join(outputDir, os.path.basename(tifPath)[:-4] + '_TOA.tif')
        math_TOA(xmlPath, tifPath, outputPath)


