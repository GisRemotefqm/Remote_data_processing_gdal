# --coding:utf-8--
import numpy as np
import pandas as pd
import os
import glob
import ReadWrite_h5 as rw
import cv2 as cv
from osgeo import gdal

def read_img(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 0, 0, im_width, im_height
    del dataset
    return im_width, im_height, im_proj, im_geotrans, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data[0])
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def seed_image(img):
    seeds = cv.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], 2000, 15, 3, 5, True)
    seeds.iterate(img, 10)  # 输入图像大小必须与初始化形状相同，迭代次数为10
    mask_seeds = seeds.getLabelContourMask()  # 获取 Mask，超像素边缘 Mask==1
    label_seeds = seeds.getLabels()  # 获取超像素标签
    number_seeds = seeds.getNumberOfSuperpixels()  # 获取超像素数目
    img_seeds = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask_seeds))


if __name__ == "__main__":
    img_path1 = r"J:\发送文件\Atmosphere\jjj\Result\Resample\230128_resample_clip.tif"
    img_path2 = r"J:\发送文件\Atmosphere\jjj\Result\Resample\231031_resample_clip.tif"

    im_width, im_height, im_proj, im_geotrans, im_data = read_img(img_path1)
    im_width2, im_height2, im_proj2, im_geotrans2, im_data2 = read_img(img_path2)
    seed_image(im_data)