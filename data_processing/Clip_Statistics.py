import numpy as np
# gdal用来处理栅格数据
from osgeo import gdal
import os
import glob
import ReadWrite_h5
import pandas as pd
# 定义栅格的读取
gdal.AllRegister()


def get_filelist(dir_path, mode='.shp'):

    dir_names = os.listdir(dir_path)
    filepath_list = []
    for name in dir_names:
        namestart = os.path.splitext(name)[0]
        nameend = os.path.splitext(name)[1]
        if nameend == mode:
            shp_path= os.path.join(dir_path, name)
            filepath_list.append(shp_path)
        if nameend == mode:
            raster_path = os.path.join(dir_path, name)
            filepath_list.append(raster_path)

    return filepath_list


def clipRaster(shp_path, raster_path, outputpath):

    ds = gdal.Warp(outputpath,  # 裁剪图像保存完整路径（包括文件名）
                   raster_path,  # 待裁剪的影像
                   format='GTiff',  # 保存图像的格式
                   cutlineDSName=shp_path,  # 矢量文件的完整路径
                   cropToCutline=True,  # 保证裁剪后影像大小跟矢量文件的图框大小一致（设置为False时，结果图像大小会跟待裁剪影像大小一样，则会出现大量的空值区域）
                   )


def read_fliename(dir_path, mode='.shp'):
    dir_names = sorted(os.listdir(dir_path))
    data_list = []
    datanames = []
    avr_list = []
    for name in dir_names:
        namestart = os.path.splitext(name)[0]
        nameend = os.path.splitext(name)[1]
        if ((mode == '.shp') and (nameend == mode)):
            shp_pathname = dir_path + '\\' + name
            print(shp_pathname)
            print(raster_path)
            clip_raster(namestart, raster_path, out_path, shp_pathname)
            if name == dir_names[-1]:
                return None

        if ((mode == '.tif') and (nameend == '.tif')):
            raster_namepath = dir_path + '\\' + name
            datanames.append(namestart)
            print(datanames)
            raster_math(raster_namepath, data_list, avr_list)
            if namestart == os.path.splitext(dir_names[-1])[0] or name == os.path.splitext(dir_names[-1])[0]:

                return data_list, avr_list, datanames


def clip_raster(shp_name, raster_path, out_path, shp_pathname):
    out_root = out_path + '\\' + shp_name + '.tif'
    print(out_root)
    ds = gdal.Warp(out_root,  # 裁剪图像保存完整路径（包括文件名）
                   raster_path,  # 待裁剪的影像
                   format='GTiff',  # 保存图像的格式
                   cutlineDSName=shp_pathname,  # 矢量文件的完整路径
                   cropToCutline=True,  # 保证裁剪后影像大小跟矢量文件的图框大小一致（设置为False时，结果图像大小会跟待裁剪影像大小一样，则会出现大量的空值区域）
                   dstNodata=np.nan)


def raster_math(raster_namepath, datalist, avr_list):

    # 获取数组
    dataset = gdal.Open(raster_namepath)
    band = dataset.GetRasterBand(1)
    dataset_arr = np.array(band.ReadAsArray())
    # 统计像元个数

    # lt10 = np.sum(dataset_arr <= -5)
    lt30 = np.sum((dataset_arr <= -0.0001) & (dataset_arr > -5))
    gt30_lt35 = np.sum((dataset_arr <= 5) & (dataset_arr > 0.001))
    gt35_lt40 = np.sum((dataset_arr <= 25) & (dataset_arr > 20))
    gt40_lt45 = np.sum((dataset_arr <= 30) & (dataset_arr > 25))
    gt45 = np.sum(dataset_arr > 5)
    list = [lt30, gt30_lt35, gt35_lt40, gt40_lt45, gt45]
    # list = [lt10, lt30, gt30_lt35, gt45]
    datalist.append(list)

    # 计算变化率
    # 将inf替换为nan
    dataset_arr[np.isinf(dataset_arr)] = np.nan

    # 将nan替换为0
    dataset_arr[np.isnan(dataset_arr)] = 0

    data_count = np.sum(dataset_arr != 0)
    data_sum = dataset_arr.sum()
    # print("data_count", data_count)
    # print("data_sum", data_sum)
    avr = (data_sum / data_count) * 100
    avr_list.append(avr)
    # print(avr)


if __name__ == '__main__':

    tifPaths = glob.glob(r".\peizhun\*.tif")
    shpPaths = glob.glob(r".\peizhun\shp\*.shp")

    for tifPath in tifPaths:
        for shpPath in shpPaths:
            print(os.path.join(r'.\peizhun\clip', os.path.basename(tifPath)[:-5] + '_clip.tif'))
            clipRaster(shpPath, tifPath, os.path.join(r'.\peizhun\clip', os.path.basename(tifPath)[:-5] + '_clip.tif'))










