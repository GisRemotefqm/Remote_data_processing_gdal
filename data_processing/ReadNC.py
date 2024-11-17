# --coding:utf-8--
import datetime
import glob
import pytz
import netCDF4 as nc
import os
import pandas as pd
import numpy as np
import math
from osgeo import gdal, osr, ogr, gdalconst
import dateutil.parser
import re
import string
from DataMath import ReadWrite_h5


def defineProjection(epsg):

    prosrs = osr.SpatialReference()
    prosrs.ImportFromEPSG(epsg)
    geosrs = prosrs.CloneGeogCS()

    return prosrs, geosrs


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


def CDF4ToTiff(inPath, groupNameList, outPath):

    Dataset = nc.Dataset(inPath)
    leftTopLon, leftTopLat, leftBottomLon, leftBottomLat = Dataset.variables['lon'][0], Dataset.variables['lat'][0],\
                                                           Dataset.variables['lon'][-1], Dataset.variables['lat'][-1]
    Xres, Yres = (leftTopLon - leftBottomLon) / len(Dataset.variables['lon']), \
                 (leftTopLat - leftBottomLat) / len(Dataset.variables['lat'])
    transform = (leftTopLon - (Xres * 0.5), Xres, 0, leftTopLat + (Yres * 0.5), 0, -Yres)
    for groupName in groupNameList:
        try:
            arr = Dataset.variables[groupName][:]

        except:
            print('not ', groupName)
            continue
        projection, geosrs = defineProjection(4326)
        write_tif(arr, projection.ExportToWkt(), transform, os.path.join(outPath, os.path.basename(inPath)[:-3] + groupName + '.tif'))


inPath = r".\20240425_00_00.nc"
print(nc.Dataset(inPath).variables['time'])
inPath = r".\20240425_00_01.nc"
print(nc.Dataset(inPath).variables['time'])
groupNameList = ['ugrd10m', 'vgrd10m', 'pressfc', 'apcpsfc']
outPath = r'.\Sentinel_5P'
CDF4ToTiff(inPath, groupNameList, outPath)


