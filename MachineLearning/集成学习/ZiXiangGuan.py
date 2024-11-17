import numpy as np
import pandas as pd
import os
from tqdm import trange, tqdm
#from haversine import haversine, haversine_vector, Unit
from osgeo import gdal
import ReadWrite_h5

# m为站点个数，lat_arr 为像素坐标, lon_arr为像素坐标, csv是做好的文件,
def generate_ps_row2(m, lat_arr, lon_arr, csv_pd, is_month):

    sites_coord = np.array(csv_pd.loc[:, ['lat', 'lon']])  # 站点坐标(lat, lon)
    # if is_month:
    #     pm25 = np.array(csv_pd['month_mean'])
    # else:
    pm25 = np.array(csv_pd['PM2.5'])

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


if __name__ == '__main__':
    path = r"K:\毕业论文\TOA拼接\moc\20200120.tif"
    dataset = gdal.Open(path)
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    imgx, imgy = dataset.RasterXSize, dataset.RasterYSize
    tif_arr = dataset.ReadAsArray(0, 0, imgx, imgy)

    height = tif_arr.shape[0]
    width = tif_arr.shape[1]

    # 生成 lon_mat, lat_mat 二维矩阵------------------------------------------
    left_lon = im_geotrans[0]
    top_lat = im_geotrans[3]
    start_lon = left_lon + 0.000
    end_lon = left_lon + (width - 1) * 0.0009 + 0.0005
    start_lat = top_lat - 0.0005
    end_lat = top_lat - (height - 1) * 0.0009 - 0.0005

    lon_arr = np.linspace(start_lon, end_lon, num=width)
    lat_arr = np.linspace(start_lat, end_lat, num=height)
    # lon_mat = np.tile(lon_arr, (height, 1)).astype('float32')  # 生成二维经度矩阵[im_height, im_width]
    # lat_mat = lat_arr.repeat(width)
    # lat_mat = np.reshape(lat_mat, (height, width)).astype('float32')  # 生成二维纬度矩阵[im_height, im_width]
    csv_path = r"J:\毕业论文\站点数据提取\231010_GF6TOA_QX_PM.csv"
    df = pd.read_csv(csv_path, encoding='gbk')
    lon_lat = df[['lon', 'lat']].to_numpy()
    # x是行号, y是列号
    x, y = ReadWrite_h5.geo2imagexy(lon_lat[:, 0], lon_lat[:, 1], im_geotrans)
    x = x.astype('int')
    y = y.astype('int')

    # 计算对应位置的中心像元位置

    center_lon, center_lat = ReadWrite_h5.imagexy2geo(dataset, x, y)
    print(center_lon, center_lat)

    unqiue_lonlat = np.array(list(set([tuple(t) for t in lon_lat])))
    num = unqiue_lonlat.shape[0]

    a, b, c = generate_ps_row2(num, center_lat, center_lon, df, True)
    df['pd'] = a
    df['pd_dist'] = b
    df['pd_num'] = c
    df.to_csv(r"J:\毕业论文\站点数据提取\231007_GF6TOA_QX_PM_自相关.csv", encoding='utf-8_sig', index=False)