import glob
import os
from osgeo import gdal
import numpy as np
import shutil
import re
def subtract_average(image1_folder, image_folder, out_folder):
    # 获取目录一中的影像文件列表
    image1_files = [os.path.join(image1_folder, file) for file in os.listdir(image1_folder) if file.endswith('.tif')]

    # 遍历目录一中的影像文件
    for image1_file in image1_files:
        # 获取第一张影像的年份和月份
        filename = os.path.basename(image1_file)
        # year = filename[5:9]
        # month_day = filename[9:13]
        year = filename[0:4]
        month = filename[5:7]
        day = filename[8:10]
        image_folder_month = glob.glob(os.path.join(image_folder, '*-' + month + '-' + day + '*.tif'))

        outpath = out_folder+"/"+filename
        # 调用处理函数
        process_image(image_folder_month, image1_file, outpath)


def process_image(image_folder, image1, outpath):
    # 打开第一张影像文件
    dataset1 = gdal.Open(image1)
    # 获取影像的行数、列数和波段数
    rows = dataset1.RasterYSize
    cols = dataset1.RasterXSize

    # 获取影像文件列表
    num_images = len(image_folder)

    # 创建一个和原始影像相同大小的数组用于存储像元值总和
    num_band_date = np.zeros((rows, cols), dtype=np.float64)
    # 遍历影像文件列表
    for image_file in image_folder:
        # 打开影像文件
        dataset = gdal.Open(image_file)
        print(dataset)
        # 读取当前影像文件的每个波段的像元值
        band_data = dataset.GetRasterBand(1).ReadAsArray()
        # 计算每一年的日异常OLR参考字段并累加
        num_band_date += band_data

        # 关闭当前影像文件
        dataset = None

    # 计算多年OLR参考字段平均值
    average_time_date = num_band_date / num_images

    # 创建一个和原始影像相同大小的新影像
    driver = dataset1.GetDriver()
    new_dataset = driver.Create(outpath, cols, rows, 1, gdal.GDT_Float32)
    # 设置新影像的空间参考信息
    new_dataset.SetGeoTransform(dataset1.GetGeoTransform())  # 写入仿射变换参数
    new_dataset.SetProjection(dataset1.GetProjection())

    # 读取第一张影像的每个波段的像元值
    img1_dataset = dataset1.GetRasterBand(1).ReadAsArray()

    # 当前影像减去平均OLR参考字段
    OLR_JuPing = img1_dataset - average_time_date

    # 将小于0的像元值赋为0
    OLR_JuPing[OLR_JuPing < 0] = 0
    OLR_JuPing = np.nan_to_num(OLR_JuPing)

    # 将新像元值保存到新波段中
    new_dataset.GetRasterBand(1).WriteArray(OLR_JuPing)


if __name__ == "__main__":
    # 指定影像文件夹路径、目录二路径和目录三路径
    # 典型事件当年影像文件路径

    image1_folder = r".\全覆盖tif\201103"
    # 所有需要计算参考字段的年份影像文件路径
    image_folder = r".\全覆盖tif\10-20"
    # 输出目录
    out_folder = r".\t2mResult\juping"

    # 调用处理函数
    subtract_average(image1_folder, image_folder, out_folder)

    print("所有影像已全部输出")
