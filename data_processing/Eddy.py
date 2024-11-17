import os
from osgeo import gdal
import numpy as np
import shutil
import glob
import re

def subtract_average(image1_folder, image_folder, out_folder):
    # 获取目录一中的影像文件列表
    image1_files = glob.glob(image1_folder)

    # 遍历目录一中的影像文件
    for image1_file in image1_files:
        # 获取第一张影像的年份和月份
        filename = os.path.basename(image1_file)
        pattern = re.compile(r'(\d{4})-(\d{2})-\d{2}')

        # 进行匹配
        match = pattern.match(filename)
        print(match[0])
        # year = match.group(1)
        # month = match.group(2)
        # day = match.group(3)

        # 构建目录二中相同月份的文件夹路径
        image_folder_month = os.path.join(image_folder, match[0] + '_t2m.tif')
        print(image_folder_month)
        exit(0)
        # 检查目录二中的文件夹是否存在
        if not os.path.exists(image_folder_month):
            print(image_folder_month+"不存在")
            continue

        outpath = out_folder+"/"+filename
        # 调用处理函数
        # get_background(image_folder_month)
        eddy_image(image1_file, outpath)


def eddy_image(image1, outpath):
    # 打开第一张影像文件
    dataset1 = gdal.Open(image1)
    # 获取影像的行数、列数和波段数
    rows = dataset1.RasterYSize
    cols = dataset1.RasterXSize

    # 创建一个和原始影像相同大小的新影像
    driver = dataset1.GetDriver()
    new_dataset = driver.Create(outpath, cols, rows, 1, gdal.GDT_Float32)
    # 设置新影像的空间参考信息
    new_dataset.SetGeoTransform(dataset1.GetGeoTransform())  # 写入仿射变换参数
    new_dataset.SetProjection(dataset1.GetProjection())

    # 读取第一张影像的每个波段的像元值
    band_data = dataset1.GetRasterBand(1).ReadAsArray()
    # 创建一个和当前波段相同大小的新波段
    new_band_data = np.zeros((rows, cols), dtype=np.float32)
    # 遍历每个像元
    for i in range(rows):
        for j in range(cols):
            # 获取上下左右四个像元的值
            up_pixel = band_data[max(i - 1, 0), j]
            down_pixel = band_data[min(i + 1, rows - 1), j]
            left_pixel = band_data[i, max(j - 1, 0)]
            right_pixel = band_data[i, min(j + 1, cols - 1)]

            # 计算当前像元值减去上下左右四个像元值
            new_pixel = (band_data[i, j] - (up_pixel + down_pixel + left_pixel + right_pixel)/4)

            # 将新像元值保存到新波段中
            new_band_data[i, j] = new_pixel

    OLR_Eddy = new_band_data
    # 将小于0的像元值赋为0
    OLR_Eddy[OLR_Eddy < 0] = 0
    OLR_Eddy = np.nan_to_num(OLR_Eddy)

    # 将新像元值保存到新波段中
    new_dataset.GetRasterBand(1).WriteArray(OLR_Eddy)


if __name__ == "__main__":
    # 指定影像文件夹路径、目录二路径和目录三路径
    # 典型事件当年影像文件路径
    image1_folder = r".\tif\t2m\2011\*.tif"
    # 所有需要计算参考字段的年份影像文件路径
    image_folder = r".\tif\t2m\其他年份"
    # 输出目录
    out_folder = r".\tif\t2m\eddy"

    # 调用处理函数
    subtract_average(image1_folder, image_folder, out_folder)

    print("所有影像已全部输出")

