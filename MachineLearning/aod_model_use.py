import os
import pandas as pd
import numpy as np
from osgeo import gdal
import joblib
from DataMath import ReadWrite_h5
from haversine import haversine, haversine_vector, Unit
from sklearn.preprocessing import StandardScaler


def ger_arr(dataset, src_dataset):
    # transform = ReadWrite_h5.get_GeoInformation(dataset)[1]
    tif_x, tif_y = ReadWrite_h5.get_RasterXY(dataset)
    src_transform = ReadWrite_h5.get_GeoInformation(dataset)[1]
    a_list = []
    b_list = []
    for i in range(tif_x):

        for i in range(tif_x):
            a = np.full(tif_y, i)
            a_list.append(a)
        for i in range(tif_y):
            b = np.full(tif_x, i)
            b_list.append(b)
    # b是纬度, a是经度
    a_list = np.array(a_list).T
    b_list = np.array(b_list)

    lon, lat = ReadWrite_h5.imagexy2geo(dataset, b_list.reshape(-1), a_list.reshape(-1))
    srcx, srcy = ReadWrite_h5.geo2imagexy(lon, lat, src_transform)
    result_list = []
    for u in range(srcx.shape[0]):

        value = ReadWrite_h5.get_RasterArr(src_dataset, rt_x=int(srcx[u]), rt_y=int(srcy[u]), lt_x=1, lt_y=1)
        value = value.tolist()[0]
        result_list.append(value)

    return np.array(result_list)

def machine_tif(model, aod_name, blh_name, r_name, sp_name, tp_name, wd_name, ws_name, t2m_name, outputpath):
    # joblib_file = machine_file
    # forest_model = joblib.load(joblib_file)

    blh = gdal.Open(blh_name)
    r = gdal.Open(r_name)
    sp = gdal.Open(sp_name)
    wd = gdal.Open(wd_name)
    ws = gdal.Open(ws_name)
    t2m = gdal.Open(t2m_name)
    dem = gdal.Open(r'J:\机器学习测试文件\dem_tif\nber_dem_1km_resample.tif')
    pop = gdal.Open(r"I:\人口数据\nber_2020_1km_resample.tif")
    df = pd.read_csv(r"C:\Users\fuqiming\Desktop\Landsat卫星表观反射率表\多点数据汇总自相关.csv")
    aod = gdal.Open(aod_name)

    aod_x, aod_y = ReadWrite_h5.get_RasterXY(aod)
    aod_arr = ReadWrite_h5.get_RasterArr(aod, aod_x, aod_y)
    aod_arr = np.nan_to_num(aod_arr)

    """
    # distance_lt_x, distance_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, distance_transform)
    # distance_lt_x, distance_lt_y = int(distance_lt_x), int(distance_lt_y)
    # distance_rb_x, distance_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, distance_transform)
    # distance_rb_x, distance_rb_y = int(distance_rb_x) - distance_lt_x, int(distance_rb_y) - distance_lt_y

    # distance_lon_l, distance_lat_l = ReadWrite_h5.imagexy2geo(distance, distance_lt_x, distance_lt_y)
    # distance_lon_y, distance_lat_r = ReadWrite_h5.imagexy2geo(distance, distance_rb_x, distance_rb_y)

    # distance_arr = ReadWrite_h5.get_RasterArr(distance, lt_x=int(distance_lt_x), lt_y=int(distance_lt_y), rt_x=int(distance_rb_y), rt_y=int(distance_rb_x))
    a_list = []
    b_list = []
    c_list = []

    # for i in range(tif_x):
    #     arg_list = []
    #     for j in range(tif_y):
    #         arg = [j + distance_lt_y, i + distance_lt_x]
    #         arg_list.append(np.array(arg))
    #     temp = np.array(arg_list)

    for i in range(tif_x):
        a = np.full(tif_y, i + distance_lt_x)
        a_list.append(a)
    for i in range(tif_y):
        b = np.full(tif_x, i + distance_lt_y)
        b_list.append(b)

    a_list = np.array(a_list).T
    b_list = np.array(b_list)


    ps_all = np.array([])
    ps_dist_all = np.array([])
    c_list = np.full((tif_x, tif_y), 43)
    for i in range(20):
        if i == 19:
            lon, lat = ReadWrite_h5.imagexy2geo(distance, b_list.reshape(-1)[int(tif_x * tif_y * 0.05) * i: ],
                                                a_list.reshape(-1)[int(tif_x * tif_y * 0.05) * i:])
        else:
            lon, lat = ReadWrite_h5.imagexy2geo(distance, b_list.reshape(-1)[int(tif_x * tif_y * 0.05) * i: int(tif_x * tif_y * 0.05) * (i+1)],
                                            a_list.reshape(-1)[int(tif_x * tif_y * 0.05) * i: int(tif_x * tif_y * 0.05) * (i+1)])
        print(lon)
        print(lat)
        ps, ps_dist, m = generate_ps_row2(43, lat, lon, df, False)
        ps_all = np.append(ps_all, ps)
        ps_all = np.nan_to_num(ps_all)
        ps_dist_all = np.append(ps_dist_all, ps_dist)
        ps_dist_all = np.nan_to_num(ps_dist_all)

    print(ps_all.shape, ps_dist_all.shape)
    """

    tif_projection, tif_transform = ReadWrite_h5.get_GeoInformation(aod)
    r_transform = ReadWrite_h5.get_GeoInformation(r)[1]
    blh_transform = ReadWrite_h5.get_GeoInformation(blh)[1]
    sp_transform = ReadWrite_h5.get_GeoInformation(sp)[1]
    t2m_transform = ReadWrite_h5.get_GeoInformation(t2m)[1]
    ws_transform = ReadWrite_h5.get_GeoInformation(ws)[1]
    wd_transform = ReadWrite_h5.get_GeoInformation(wd)[1]
    dem_transform = ReadWrite_h5.get_GeoInformation(dem)[1]
    pop_transform = ReadWrite_h5.get_GeoInformation(pop)[1]

    # 左上
    lt_lon, lt_lat = ReadWrite_h5.imagexy2geo(aod, 0, 0)
    # # 左下
    # lb_lon, lb_lat = ReadWrite_h5.imagexy2geo(tif, 0, tif_y)
    # # 右上
    # rt_lon, rt_lat = ReadWrite_h5.imagexy2geo(tif, tif_x, 0)
    # # 右下
    rb_lon, rb_lat = ReadWrite_h5.imagexy2geo(aod, aod_x, aod_y)
    print(lt_lon, lt_lat, rb_lon, rb_lat)

    r_lt_x, r_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, r_transform)
    r_lt_x, r_lt_y = int(r_lt_x), int(r_lt_y)
    r_rb_x, r_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, r_transform)
    r_rb_x, r_rb_y = int(r_rb_x) - r_lt_x, int(r_rb_y) - r_lt_y

    sp_lt_x, sp_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, sp_transform)
    sp_lt_x, sp_lt_y = int(sp_lt_x), int(sp_lt_y)
    sp_rb_x, sp_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, sp_transform)
    sp_rb_x, sp_rb_y = int(sp_rb_x) - sp_lt_x, int(sp_rb_y) - sp_lt_y

    wd_lt_x, wd_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, wd_transform)
    wd_lt_x, wd_lt_y = int(wd_lt_x), int(wd_lt_y)
    wd_rb_x, wd_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, wd_transform)
    wd_rb_x, wd_rb_y = int(wd_rb_x) - wd_lt_x, int(wd_rb_y) - wd_lt_y

    ws_lt_x, ws_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, ws_transform)
    ws_lt_x, ws_lt_y = int(ws_lt_x), int(ws_lt_y)
    ws_rb_x, ws_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, ws_transform)
    ws_rb_x, ws_rb_y = int(ws_rb_x) - ws_lt_x, int(ws_rb_y) - ws_lt_y

    blh_lt_x, blh_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, blh_transform)
    blh_lt_x, blh_lt_y = int(blh_lt_x), int(blh_lt_y)
    blh_rb_x, blh_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, blh_transform)
    blh_rb_x, blh_rb_y = int(blh_rb_x) - blh_lt_x, int(blh_rb_y) - blh_lt_y

    t2m_lt_x, t2m_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, t2m_transform)
    t2m_lt_x, t2m_lt_y = int(t2m_lt_x), int(t2m_lt_y)
    t2m_rb_x, t2m_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, t2m_transform)
    t2m_rb_x, t2m_rb_y = int(t2m_rb_x) - t2m_lt_x, int(t2m_rb_y) - t2m_lt_y

    dem_lt_x, dem_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, dem_transform)
    dem_lt_x, dem_lt_y = int(dem_lt_x), int(dem_lt_y)
    dem_rb_x, dem_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, dem_transform)
    dem_rb_x, dem_rb_y = int(dem_rb_x) - dem_lt_x, int(dem_rb_y) - dem_lt_y

    pop_lt_x, pop_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, pop_transform)
    pop_lt_x, pop_lt_y = int(pop_lt_x), int(pop_lt_y)
    pop_rb_x, pop_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, pop_transform)
    pop_rb_x, pop_rb_y = int(pop_rb_x) - pop_lt_x, int(pop_rb_y) - pop_lt_y

    r_arr = ReadWrite_h5.get_RasterArr(r, lt_x=int(r_lt_x), lt_y=int(r_lt_y), rt_x=int(r_rb_y), rt_y=int(r_rb_x))
    sp_arr = ReadWrite_h5.get_RasterArr(sp, lt_x=int(sp_lt_x), lt_y=int(sp_lt_y), rt_x=int(sp_rb_y), rt_y=int(sp_rb_x))
    wd_arr = ReadWrite_h5.get_RasterArr(wd, lt_x=int(wd_lt_x), lt_y=int(wd_lt_y), rt_x=int(wd_rb_y), rt_y=int(wd_rb_x))
    ws_arr = ReadWrite_h5.get_RasterArr(ws, lt_x=int(ws_lt_x), lt_y=int(ws_lt_y), rt_x=int(ws_rb_y), rt_y=int(ws_rb_x))
    t2m_arr = ReadWrite_h5.get_RasterArr(t2m, lt_x=int(t2m_lt_x), lt_y=int(t2m_lt_y), rt_x=int(t2m_rb_y),
                                         rt_y=int(t2m_rb_x))
    blh_arr = ReadWrite_h5.get_RasterArr(blh, lt_x=int(blh_lt_x), lt_y=int(blh_lt_y), rt_x=int(blh_rb_y),
                                         rt_y=int(blh_rb_x))
    dem_arr = ReadWrite_h5.get_RasterArr(dem, lt_x=int(dem_lt_x), lt_y=int(dem_lt_y), rt_x=int(dem_rb_y),
                                         rt_y=int(dem_rb_x))
    pop_arr = ReadWrite_h5.get_RasterArr(pop, lt_x=int(pop_lt_x), lt_y=int(pop_lt_y), rt_x=int(pop_rb_y),
                                         rt_y=int(pop_rb_x))
    # pop_arr = np.log((pop_arr + 1))
    print(aod_arr.reshape(-1).shape)
    print(blh_arr.reshape(-1).shape)
    print(r_arr.reshape(-1).shape)
    print(sp_arr.reshape(-1).shape)
    print(t2m_arr.reshape(-1).shape)
    print(wd_arr.reshape(-1).shape)
    print(ws_arr.reshape(-1).shape)
    print(pop_arr[0].reshape(-1).shape)
    aod_arr = aod_arr * 1000
    x_test = np.vstack((blh_arr.reshape(-1), r_arr.reshape(-1), sp_arr.reshape(-1), t2m_arr.reshape(-1),
                        wd_arr.reshape(-1), ws_arr.reshape(-1), dem_arr[0].reshape(-1), aod_arr.reshape(-1),
                        pop_arr.reshape(-1)))
    scaler = StandardScaler()
    print(x_test.shape)
    x_test = x_test.T

    print(x_test.shape)
    # x_test = scaler.fit_transform(x_test)

    pm_result = model.predict(x_test)
    pm_result = np.resize(pm_result, (aod_y, aod_x))
    print(aod_arr.shape)
    pm_result[aod_arr[0] == 0] = 0

    ReadWrite_h5.write_tif(pm_result, tif_projection, tif_transform, outputpath)


if __name__ == '__main__':
    tif_path = r'J:\AODinversion\AOD的TOA数据\月平均\Landsat\nber'
    blh_path = r'I:\1KM重采样\尼泊尔\blh'
    r_path = r'I:\1KM重采样\尼泊尔\r'
    sp_path = r'I:\1KM重采样\尼泊尔\sp'
    tp_path = r'I:\1KM重采样\尼泊尔\tp'
    wd_path = r'I:\1KM重采样\尼泊尔\wd'
    ws_path = r'I:\1KM重采样\尼泊尔\ws'
    t2m_path = r'I:\1KM重采样\尼泊尔\t2m'
    output = r'J:\机器学习测试文件\result\xgboost_nber_aod_2020'
    machine_path = '日平均/日平均模型结果/xgboost模型/xboost_model_mul_all_aod_tun_非对数人口.m'
    model = joblib.load(machine_path)

    for temp in os.listdir(tif_path):

        if os.path.splitext(temp)[1] == '.tif':

            tif_name = os.path.join(tif_path, temp)
            date = temp.split('_')[0]
            blh_name = os.path.join(blh_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_blh_resample.tif')
            r_name = os.path.join(r_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_r_resample.tif')
            sp_name = os.path.join(sp_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_sp_resample.tif')
            tp_name = os.path.join(tp_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_tp_resample.tif')
            wd_name = os.path.join(wd_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_wd_resample.tif')
            ws_name = os.path.join(ws_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_ws_resample.tif')
            t2m_name = os.path.join(t2m_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_t2m_resample.tif')
            outputpath = os.path.join(output, os.path.splitext(temp)[0] + '_machine_aod.tif')

            machine_tif(model, tif_name, blh_name, r_name, sp_name, tp_name, wd_name, ws_name, t2m_name, outputpath)