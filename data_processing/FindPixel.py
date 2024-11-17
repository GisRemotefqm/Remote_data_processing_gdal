import numpy as np
import os

import pandas as pd

import ReadWrite_h5
import MathAOD.MathAod as md
import Mergeh5
import re
import math
import osgeo.osr as osr


def findinputdir(inputpath, ob_time):
    """
    根据AER数据与DPC日期进行匹配，返回日期列表
    :param inputpath: 存放DPC数据的路径
    :param ob_time: 值为AER数据[年份，具体时间]{已经经过处理}
    :return: 日期列表
    """
    inputdir = os.listdir(inputpath)
    start_time = ob_time[0].tolist()
    inputdir = list(set(inputdir) & set(start_time))

    return inputdir


def math_polorar(h5_path, dirs, lon, lat):
    poloar_list = []
    date_list = []
    sc_list = []
    sinpro, singeo = ReadWrite_h5.defineSin()
    # prosrs, geosrs = ReadWrite_h5.defineSRS(4326)
    for dir in dirs:

        inputdir = os.path.join(h5_path, dir)
        inputpaths = os.listdir(inputdir)

        for inputpath in inputpaths:

            inputname = os.path.join(inputdir, inputpath)
            namestart, nameend = os.path.splitext(inputpath)
            re_result = re.findall(r'B865$', namestart)
            if re_result != []:
                print(inputname)

                h5_filelist = ReadWrite_h5.get_h5filelist(inputname)

                Lon = ReadWrite_h5.get_h5Dataset(h5_filelist, 'Longitude')
                Lat = ReadWrite_h5.get_h5Dataset(h5_filelist, 'Latitude')
                Lon_arr = ReadWrite_h5.get_RasterArr(Lon)
                Lon_arr = Lon_arr[0]
                Lat_arr = ReadWrite_h5.get_RasterArr(Lat)
                Lat_arr = Lat_arr[0]
                Idataset = ReadWrite_h5.get_h5Dataset(h5_filelist, 'I865P')
                Idata_arr = ReadWrite_h5.get_RasterArr(Idataset)

                transform1 = ReadWrite_h5.get_TransForm(Lon_arr, Lat_arr, sinpro, singeo)
                prolon, prolat = ReadWrite_h5.lonlatToGeo(39.93333, 116.31667, sinpro, singeo)

                x2, y2 = ReadWrite_h5.geo2imagexy(prolon, prolat, transform1)
                print(x2, y2)
                print('x2: {}, y2: {}, x2type: {}, y2tpye: {}'.format(x2, y2, type(x2), type(y2)))
                print(Idata_arr[:, int(y2), int(x2)])
                Ifill_value = Mergeh5.get_fillvalue(h5_filelist, datasetname='I865P')
                if abs(Idata_arr[0, int(y2), int(x2)]) == Ifill_value:
                    print('说明这个地方是妹有影像的, 过')
                    continue

                sza_scale_factor, sza_add_offset = Mergeh5.get_h5Attribute(inputname, 'Data_Fields', 'Sol_Zen_Ang')
                vza_scale_factor, vza_add_offset = Mergeh5.get_h5Attribute(inputname, 'Data_Fields', 'View_Zen_Ang')
                phiv_scale_factor, phiv_add_offset = Mergeh5.get_h5Attribute(inputname, 'Data_Fields', 'View_Azim_Ang')
                phis_scale_factor, phis_add_offset = Mergeh5.get_h5Attribute(inputname, 'Data_Fields', 'Sol_Azim_Ang')
                sza_ob_geo = Mergeh5.math_ObserveGeometry(sza_add_offset, sza_scale_factor, h5_filelist, 'Sol_Zen_Ang')
                vza_ob_geo = Mergeh5.math_ObserveGeometry(vza_scale_factor, vza_add_offset, h5_filelist, 'View_Zen_Ang')
                phiv_ob_geo = Mergeh5.math_ObserveGeometry(phiv_scale_factor, phiv_add_offset, h5_filelist, 'View_Azim_Ang')
                phis_ob_geo = Mergeh5.math_ObserveGeometry(phis_scale_factor, phis_add_offset, h5_filelist, 'Sol_Azim_Ang')

                sza_arr = sza_ob_geo['Sol_Zen_Ang']
                vza_arr = vza_ob_geo['View_Zen_Ang']
                phiv_arr = phiv_ob_geo['View_Azim_Ang']
                phis_arr = phis_ob_geo['Sol_Azim_Ang']

                sc_data = math_sc(sza_arr[:, int(y2), int(x2)], phis_arr[:, int(y2), int(x2)],
                                  vza=vza_arr[:, int(y2), int(x2)], phiv=phiv_arr[:, int(y2), int(x2)])

                # 读取GF-5数据中的数据值
                Qdataset = ReadWrite_h5.get_h5Dataset(h5_filelist, 'Q865P')
                Udataset = ReadWrite_h5.get_h5Dataset(h5_filelist, 'U865P')

                Qdata_arr = ReadWrite_h5.get_RasterArr(Qdataset)
                Udata_arr = ReadWrite_h5.get_RasterArr(Udataset)

                result = (math.pi * np.sqrt(np.square(Qdata_arr[:, int(y2), int(x2)]) +
                                            np.square(Udata_arr[:, int(y2), int(x2)]))
                          / Idata_arr[:, int(y2), int(x2)] * np.cos(sza_arr[:, int(y2), int(x2)]))
                result = np.array(result)

                print('散射角为：', sc_data)
                print('对应的地表偏振反射率为：', result)
                sc_list.append(sc_data)
                poloar_list.append(result)
                date_list.append(dir)



    jiaodu = ['角度' + str(i + 1) for i in range(12)]

    poloar = pd.DataFrame(np.array(poloar_list), index=date_list, columns=jiaodu)


    return poloar, np.array(sc_list)


def math_sc(sza, phis, vza, phiv):
    """

    :param SZA: 太阳天顶角
    :param phis: 太阳方位角
    :param VZA: 观测天顶角
    :param phiv: 观测方位角
    :return: 散射角
    """
    SC = []
    sza = np.deg2rad(sza)
    vza = np.deg2rad(vza)
    phiv = np.deg2rad(phiv)
    phis = np.deg2rad(phis)
    raa = abs(phiv - phis)
    for i in range(raa.shape[0]):
        if raa[i] > 180:
            raa[i] = 360 - raa[i]
        raa[i] = 180 - raa[i]
        sc = np.arccos(-np.cos(vza[i]) *
                       np.cos(sza[i]) + np.sin(vza[i]) *
                       np.sin(sza[i]) * np.cos(raa[i]))
        # 将弧度转化为度
        SC.append(np.rad2deg(sc))

    return np.array(SC)


def read_modis(modis_inputpath, lon, lat):
    # 读取MODIS产品中的数据值
    tifdata = ReadWrite_h5.get_tifDataset(modis_inputpath)

    projection, transform = ReadWrite_h5.get_GeoInformation(tifdata)
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(projection)
    geosrs = prosrs.CloneGeogCS()
    prolon, prolat = ReadWrite_h5.lonlatToGeo(lon, lat, prosrs, geosrs)
    x, y = ReadWrite_h5.geo2imagexy(prolon, prolat, transform)
    print('x: {}, y: {}, xtype: {}, ytpye: {}'.format(x, y, type(x), type(y)))
    tif_arr = ReadWrite_h5.get_RasterArr(tifdata)
    print('tif.shape: ', tif_arr.shape)
    print('对应点值：', tif_arr[0, int(y), int(x)])

    tif_arr = tif_arr[0, int(y - 3): int(y + 4), int(x - 3): int(x + 4)]

    flatten_tif_arr = tif_arr.flatten()
    list_arr = flatten_tif_arr.tolist()
    list_arr = list(set(list_arr))
    for value in list_arr:
        num = np.count_nonzero(tif_arr == value)
        if (num / tif_arr.size) > 0.75:
            print('该像元对应的GF-5像元为纯像元，其值为{}'.format(value))
            return True
    else:
        print('该像元不是纯像元')
        return False

if __name__ == '__main__':
    modis_inputpath = r'I:\MODIS_AOD\2020Majority_Land_Cover_Type_1.tiff'
    aod_inputpath = r'I:\MODIS_AOD\20200101_20201231_Beijing-CAMS\20200101_20201231_Beijing-CAMS.csv'
    lon = 116.31667
    lat = 39.93333
    h5_path = r'I:\DPC日期'
    aod_name = 'AOD_870nm'
    ob_time = md.find_AOD(aod_inputpath, aod_name, 0.1)
    dirs = findinputdir(h5_path, ob_time)


    read_modis(modis_inputpath, lon, lat)
    poloar, sc = math_polorar(h5_path, dirs, lat, lon)
    sc = pd.DataFrame(sc)
    sc.to_csv('sc_result.csv')
    poloar.to_csv('result.csv')


