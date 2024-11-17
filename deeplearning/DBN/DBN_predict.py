import numpy as np
import torch
import os
import operator
from osgeo import gdal
from DBN import DBN
from tqdm import trange, tqdm
from haversine import haversine, haversine_vector, Unit
import pandas as pd
from math import sin, asin, cos, radians, fabs, sqrt, pi

Radius = 6371.009  # 地球半径, 单位km
h = 50  # window size, 单位为km
device = torch.device('cuda:0')


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


def match_file_name(day, file_path, file_type):
    """
    根据日期 匹配tiff文件, csv文件
    """
    tiff_list = get_file_list(file_path, file_type)
    for t_name, t_path in tiff_list:
        if operator.contains(t_name, day):
            return t_path


def get_file_path(day, f_dict):
    """
    根据 f_dict 里的文件夹路径, 找到与 day 匹配的文件 path
    """

    aod_path = match_file_name(day, f_dict['aod_path'], 'tiff')
    blh_path = match_file_name(day, f_dict['blh_path'], 'tiff')
    rh_path = match_file_name(day, f_dict['rh_path'], 'tiff')
    t_path = match_file_name(day, f_dict['t_path'], 'tiff')
    wind_path = match_file_name(day, f_dict['wind_path'], 'tiff')
    wd_path = match_file_name(day, f_dict['wd_path'], 'tiff')
    sp_path = match_file_name(day, f_dict['sp_path'], 'tiff')
    if aod_path:
        print(aod_path.split('\\')[-1])
    print(blh_path.split('\\')[-1])
    print(rh_path.split('\\')[-1])
    print(t_path.split('\\')[-1])
    print(wind_path.split('\\')[-1])
    print(wd_path.split('\\')[-1])
    print(sp_path.split('\\')[-1])

    ndvi_path = match_file_name(day[0:6], f_dict['ndvi_path'], 'tiff')
    lct_path = f_dict['lct_file']
    pop_path = f_dict['pop_file']
    dem_path = f_dict['dem_file']
    print(ndvi_path.split('\\')[-1])

    return aod_path, blh_path, rh_path, t_path, wind_path, wd_path, sp_path, ndvi_path, lct_path, pop_path, dem_path


def get_everyday(year, begin_date, end_date):
    if isinstance(year, int) is False:
        raise FileNotFoundError("the type of year must be int!")
    everyday = []
    for i in range(1, 13):
        months = 100 + i
        if i == 2:
            if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
                days = 29
            else:
                days = 28
        elif i in {1, 3, 5, 7, 8, 10, 12}:
            days = 31
        else:
            days = 30
        for j in range(101, 101+days):
            y = str(year)
            month = str(months)[1:]
            day = str(j)[1:]
            everyday.append(y + month + day)

    assert begin_date in everyday and end_date in everyday, "begin_date or end_date is wrong!"
    begin_index = everyday.index(begin_date)
    end_index = everyday.index(end_date)
    return everyday[begin_index:end_index + 1]


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


def get_max(csv_path, ps_name, ps_dist_num):

    csv_data = pd.read_csv(filepath_or_buffer=csv_path)
    valid_data = csv_data.dropna(axis=0)  # 剔除nan

    aod_nd = valid_data['aod_13'].to_numpy(dtype='float32')
    blh_nd = valid_data['blh_13'].to_numpy(dtype='float32')
    rh_nd = valid_data['rh_13'].to_numpy(dtype='float32')
    t_nd = valid_data['t_13'].to_numpy(dtype='float32')
    ws_nd = valid_data['ws_13'].to_numpy(dtype='float32')
    wd_nd = valid_data['wd_13'].to_numpy(dtype='float32')
    sp_nd = valid_data['sp_13'].to_numpy(dtype='float32')
    ndvi_nd = valid_data['ndvi_13'].to_numpy(dtype='float32')
    lct_nd = valid_data['lct_13'].to_numpy(dtype='float32')
    lct_nd = np.round(lct_nd)
    pop_nd = valid_data['pop_13'].to_numpy(dtype='float32')
    dem_nd = valid_data['dem_13'].to_numpy(dtype='float32')
    ps_nd = valid_data[ps_name].to_numpy(dtype='float32')
    ps_dist_nd = valid_data[ps_dist_num].to_numpy(dtype='float32')
    # ps_num_nd = valid_data['ps_num'].to_numpy(dtype='float32')

    contents_data = np.stack((
        aod_nd, blh_nd, rh_nd, t_nd, ws_nd, wd_nd, sp_nd, ndvi_nd, lct_nd, pop_nd, dem_nd, ps_nd, ps_dist_nd), axis=-1)

    max_v = np.amax(contents_data, axis=0)  # 每个特征的最大值, 数据归一化的分母

    return max_v


def load_img_data(aod_fp, blh_fp, rh_fp, t_fp, wind_fp, wd_fp, sp_fp, ndvi_fp, lct_fp, pop_fp, dem_fp):
    """
    根据文件 path, 读取v张图像, 返回一个 (im_height * im_width * v) 维度的数据
    """

    col_num = [gdal.Open(aod_fp).RasterXSize, gdal.Open(blh_fp).RasterXSize, gdal.Open(rh_fp).RasterXSize,
               gdal.Open(t_fp).RasterXSize, gdal.Open(wind_fp).RasterXSize, gdal.Open(wd_fp).RasterXSize,
               gdal.Open(sp_fp).RasterXSize, gdal.Open(ndvi_fp).RasterXSize, gdal.Open(lct_fp).RasterXSize,
               gdal.Open(pop_fp).RasterXSize, gdal.Open(dem_fp).RasterXSize]

    row_num = [gdal.Open(aod_fp).RasterYSize, gdal.Open(blh_fp).RasterYSize, gdal.Open(rh_fp).RasterYSize,
               gdal.Open(t_fp).RasterYSize, gdal.Open(wind_fp).RasterYSize, gdal.Open(wd_fp).RasterYSize,
               gdal.Open(sp_fp).RasterYSize, gdal.Open(ndvi_fp).RasterYSize, gdal.Open(lct_fp).RasterYSize,
               gdal.Open(pop_fp).RasterYSize, gdal.Open(dem_fp).RasterYSize]

    im_width = min(col_num)
    im_height = min(row_num)

    aod_data = gdal.Open(aod_fp).ReadAsArray(0, 0, im_width, im_height)
    blh_data = gdal.Open(blh_fp).ReadAsArray(0, 0, im_width, im_height)
    rh_data = gdal.Open(rh_fp).ReadAsArray(0, 0, im_width, im_height)
    t_data = gdal.Open(t_fp).ReadAsArray(0, 0, im_width, im_height)
    wind_data = gdal.Open(wind_fp).ReadAsArray(0, 0, im_width, im_height)
    wd_data = gdal.Open(wd_fp).ReadAsArray(0, 0, im_width, im_height)
    sp_data = gdal.Open(sp_fp).ReadAsArray(0, 0, im_width, im_height)

    ndvi_data = gdal.Open(ndvi_fp).ReadAsArray(0, 0, im_width, im_height)
    lct_data = gdal.Open(lct_fp).ReadAsArray(0, 0, im_width, im_height)
    pop_data = gdal.Open(pop_fp).ReadAsArray(0, 0, im_width, im_height)
    pop_data = pop_data.astype(np.float32)
    dem_data = gdal.Open(dem_fp).ReadAsArray(0, 0, im_width, im_height)
    dem_data = dem_data.astype(np.float32)

    aod_data[np.where(aod_data == 0.0)] = np.nan
    blh_data[np.where(blh_data <= -10000.0)] = np.nan
    rh_data[np.where(rh_data <= -10000.0)] = np.nan
    t_data[np.where(t_data <= -10000.0)] = np.nan
    wind_data[np.where(wind_data <= -10000.0)] = np.nan
    wd_data[np.where(wd_data <= -10000.0)] = np.nan
    sp_data[np.where(sp_data <= -10000.0)] = np.nan
    ndvi_data[np.where(ndvi_data <= -10000.0)] = np.nan
    lct_data[np.where(lct_data <= -10000.0)] = np.nan
    pop_data[np.where(pop_data <= -10000.0)] = np.nan
    dem_data[np.where(dem_data <= -10000.0)] = np.nan

    aod_3d = np.reshape(aod_data, (im_height, im_width, 1))
    ndvi_3d = np.reshape(ndvi_data, (im_height, im_width, 1))
    lct_3d = np.reshape(lct_data, (im_height, im_width, 1))
    pop_3d = np.reshape(pop_data, (im_height, im_width, 1))
    dem_3d = np.reshape(dem_data, (im_height, im_width, 1))

    blh_3d = np.reshape(blh_data, (im_height, im_width, 1))
    rh_3d = np.reshape(rh_data, (im_height, im_width, 1))
    t_3d = np.reshape(t_data, (im_height, im_width, 1))
    wind_3d = np.reshape(wind_data, (im_height, im_width, 1))
    wd_3d = np.reshape(wd_data, (im_height, im_width, 1))
    sp_3d = np.reshape(sp_data, (im_height, im_width, 1))
    # aod_nd, blh_nd, rh_nd, t_nd, ws_nd, wd_nd, sp_nd, ndvi_nd, lct_nd, pop_nd, dem_nd, ps_nd

    img_data = np.concatenate((aod_3d, blh_3d, rh_3d, t_3d, wind_3d, wd_3d, sp_3d,
                               ndvi_3d, lct_3d, pop_3d, dem_3d), axis=2)  # 合并为3维数组

    return img_data


def predict(pre_train_path, checkpoint_path, input_array):

    dbn = DBN(input_array.shape[1], dbn_layers, savefile=pre_train_path)
    model = dbn.load_pretrained_model()  # dbn已经过训练, 直接将 savefile 加载到model

    model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        x = torch.from_numpy(input_array).to(device)
        y = model(x).cpu().detach().numpy().flatten()
        return y


def generate_ps_row2(m, lat_arr, lon_arr, csv_pd, is_month):

    sites_coord = np.array(csv_pd.loc[:, ['lat', 'lon']])  # 站点坐标(lat, lon)
    if is_month:
        pm25 = np.array(csv_pd['month_mean'])
    else:
        pm25 = np.array(csv_pd['daily_mean'])

    pixel_coord = np.stack((lat_arr, lon_arr), axis=-1)  # 像素点坐标(lat, lon)

    dist_mat = haversine_vector(pixel_coord, sites_coord, Unit.KILOMETERS, comb=True)
    dist_mat = dist_mat.T  # dist_mat: (像素个数, 站点个数), 像素点到每个站点的距离
    pixel_num = dist_mat.shape[0]

    pm25_mat = np.tile(pm25, (pixel_num, 1))  # pm25_mat: (像素个数, 站点个数)

    # 李同文求w(固定站点个数)-------------------------------------
    m_idx = np.argpartition(dist_mat, axis=1, kth=m)[:, :m]
    pixel_idx = np.arange(0, pixel_num).repeat(m)
    site_idx = m_idx.flatten()
    d = dist_mat[pixel_idx, site_idx].reshape((pixel_num, m))
    pm25_obs = pm25_mat[pixel_idx, site_idx].reshape((pixel_num, m))

    w_arr = 1 / (d * d)  # w_arr: (像素个数, m)
    w_pm25 = w_arr * pm25_obs  # w_pm25: (像素个数, m)
    sum_w_pm25 = np.sum(w_pm25, axis=1)  # (像素个数,)
    sum_w = np.sum(w_arr, axis=1)  # (像素个数,)

    ps_arr = sum_w_pm25 / sum_w
    ps_dist_arr = np.min(dist_mat, axis=1)  # min_dist: (像素个数,)

    return ps_arr, ps_dist_arr, m


def dbn_predict(day_list, f_dict, pm25_p, max_features, pre_train_p, check_p, out_p, name, month):

    days_num = len(day_list)

    dbn = DBN(13, dbn_layers, savefile=pre_train_p)
    model = dbn.load_pretrained_model()  # dbn已经过训练, 直接将 savefile 加载到model

    model.to(device)
    checkpoint = torch.load(check_p)
    model.load_state_dict(checkpoint['model'])

    year_arr = np.array([2016, 2017, 2018, 2019, 2020])
    mean_year = np.mean(year_arr)
    std_year = np.std(year_arr)

    for dd in range(0, days_num):

        day = day_list[dd]
        output_path = out_p + '/' + day + name + '.tiff'

        aod_p2, blh_p2, rh_p2, t_p2, wind_p2, wd_p2, sp_p2, ndvi_p2, lct_p2, pop_p2, dem_p2 = \
            get_file_path(day, f_dict)

        if not aod_p2:
            continue

        img_day = \
            load_img_data(aod_p2, blh_p2, rh_p2, t_p2, wind_p2, wd_p2, sp_p2, ndvi_p2, lct_p2, pop_p2, dem_p2)

        aod = gdal.Open(aod_p2)
        im_geotrans = aod.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        im_proj = aod.GetProjection()  # 地图投影信息，字符串表示

        height = img_day.shape[0]
        width = img_day.shape[1]
        output_img = np.zeros([height, width], dtype='float32') + np.nan

        pm25_path = match_file_name(day, pm25_p, 'csv')
        print(pm25_path.split('\\')[-1])
        csv_pd = pd.read_csv(pm25_path)

        if csv_pd.empty:
            continue

        for i in tqdm(range(0, height)):
            x_row = img_day[i, :, :]
            z_index = np.where(np.isnan(x_row))
            zero_index = np.unique(z_index[0])  # 提取出含有nan的列的索引
            if zero_index.shape[0] == width:
                continue

            all_index = np.arange(0, width)
            non_zero_index = np.setdiff1d(all_index, zero_index)  # 去掉有nan的列索引, 保留其余索引
            input_array = x_row[non_zero_index]



            input_array = input_array/max_features

            if month:
                y = int(day[0:4])
                m = int(day[4:6])
                pt = cos(2 * pi * m / 12) + np.zeros((non_zero_index.shape[0], 1))
                year_code = (y - mean_year) / std_year + np.zeros((non_zero_index.shape[0], 1))
                input_array = np.append(input_array, pt, axis=1)
                input_array = np.append(input_array, year_code, axis=1)

            input_array = input_array.astype('float32')

            with torch.no_grad():
                x = torch.from_numpy(input_array).to(device)
                y = model(x).cpu().detach().numpy().flatten()
                output_img[i, non_zero_index] = y

        create_tiff(output_path, output_img, im_geotrans, im_proj)


if __name__ == '__main__':

    max_vv = get_max(r'E:\China_PM25_csv\China_24hours_month_2016_2020_gt_ps_bilinear2_remove_2628_2629.csv',
                     'ps_li_10', 'ps_dist_10')

    dbn_layers = [15, 15, 1]
    pretrained_file = r'D:\Projects_Python\220322_DBN\2016_2020_month_test\2016_2020_china_100.pt'
    checkpoint_file = r'D:\Projects_Python\220322_DBN\2016_2020_month_test\checkpoints\checkpoint_100.pth'

    pm25_csv_p = r'E:\China_2016\2016_PM25_month_mean'
    output_p = 'E:/China_2018/dbn_predict_10'

    ml3 = ['201801', '201802', '201803', '201804', '201805', '201806',
           '201807', '201808', '201809', '201810', '201811', '201812']

    file_dict3 = {
        'aod_path': 'H:/china_2018_MCD19A2/a7',
        'blh_path': 'H:/2018/step4/2018_blh_4',
        'rh_path': 'H:/2018/step4/2018_r_4',
        't_path': 'H:/2018/step4/2018_t2m_4',
        'wind_path': 'H:/2018/step4/2018_wind_4',
        'wd_path': 'H:/2018/step4/2018_wd_4',
        'sp_path': 'H:/2018/step4/2018_sp_4',

        'ndvi_path': 'E:/MOD13A3/step4',
        'lct_file': 'E:/MCD12Q1/step3/MCD12Q1.A2018001_mask.tiff',
        'pop_file': 'E:/landscan-global/resample/landscan-global-2018_mask_resample.tiff',
        'dem_file': 'E:/GMTED2010/GMTED2010_mask_resample.tiff'
    }

    dbn_predict(
        ml3, file_dict3, pm25_csv_p, max_vv, pretrained_file, checkpoint_file, output_p,
        name='_ps_10', month=True)

