# --coding:utf-8--
import pywt
from osgeo import gdal
import numpy as np


def tif2matrix(tif_arr, level):
    w = 'db1'  # 小波基类型
    # w = 'cmor'
    temp = tif_arr
    # for i in range(level - 1):
    #     cA, (cH, cV, cD) = pywt.dwt2(temp, w)
    #     cH = np.zeros_like(cH)
    #     cV = np.zeros_like(cV)
    #     cD = np.zeros_like(cD)
    #
    #     temp = cA
    #


        # 执行多级小波分解
    coeffs = pywt.wavedec2(tif_arr, w, level=level)
    # 获取低频部分（近似系数）
    cA = coeffs[0]
    # reconstructed_low_freq = pywt.waverec2([cA], w)
    print(cA.shape)
    result = np.resize(cA, (tif_arr.shape[0], tif_arr.shape[1]))
    return cA


def write_tif(data_list, projection, transform, outputPath):

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


def mutilWaveletsChange(tifPath, Nodata, level, outPath='', tempFlag=0, tempPath=''):

    """

    :param tifPath: 遥感影像路径
    :param Nodata: 遥感影像无效值
    :param level: 小波变换级数
    :param tempFlag: 是否存储中间影像
    :param tempDataset: 中间影像路径
    :param outPath: 输出文件
    :return:
    """

    dataset = gdal.Open(tifPath)
    projection = dataset.GetProjection()
    transform = dataset.GetGeoTransform()
    imgx, imgy = dataset.RasterXSize, dataset.RasterYSize
    tifArr = dataset.ReadAsArray(0, 0, imgx, imgy)
    tifArr[tifArr == Nodata] = 0

    x_res = transform[1] * 2 ** level
    y_res = transform[5] * 2 ** level
    print(transform)
    new_transform = (transform[0], x_res, transform[2], transform[3], transform[4], y_res)
    print(new_transform)
    reconstructed_tif_arr = tif2matrix(tifArr, level=level)
    print(reconstructed_tif_arr.shape)


    def createTifDriver(driverName, dataArr, tempPath=''):

        if (driverName == 'GTiff') & (tempPath == ''):
            raise Exception('outPath undefined')

        if len(dataArr.shape) > 2:
            driver = gdal.GetDriverByName(driverName)
            mem_dataset = driver.Create(tempPath, dataArr.shape[2], dataArr.shape[1],
                                        dataArr.shape[0], gdal.GDT_Float32)
            mem_dataset.SetGeoTransform(new_transform)
            mem_dataset.SetProjection(projection)

            for i in range(dataArr.shape[0]):
                mem_dataset.GetRasterBand(i + 1).WriteArray(dataArr[i])

        else:
            driver = gdal.GetDriverByName(driverName)
            mem_dataset = driver.Create(tempPath, dataArr.shape[1], dataArr.shape[0],
                                        1, gdal.GDT_Float32)
            mem_dataset.SetGeoTransform(new_transform)
            mem_dataset.SetProjection(projection)
            mem_dataset.GetRasterBand(1).WriteArray(dataArr)

        return mem_dataset


    outBounds = (*CoordTransf(0, 0, transform), *CoordTransf(imgx, imgy, transform))
    print(outBounds)
    if tempFlag == 0:
        print('果真是你吗')
        mem_dataset = createTifDriver('MEM', reconstructed_tif_arr)
        gdal.Warp(outPath, mem_dataset, resampleAlg=gdal.GRA_Bilinear, outputBounds=outBounds,
                  srcNodata=0, dstNodata=-1, xRes=transform[1], yRes=transform[1])

    elif tempFlag == 1:
        mem_dataset = createTifDriver('MEM', reconstructed_tif_arr, tempPath)
        write_tif(reconstructed_tif_arr, projection, new_transform, tempPath)

        gdal.Warp(outPath, mem_dataset, resampleAlg=gdal.GRA_Bilinear, outputBounds=outBounds,
                  srcNodata=0, dstNodata=-1, xRes=transform[1], yRes=transform[1])
    else:
        raise Exception('出错了')


def CoordTransf(Xpixel,Ypixel,GeoTransform):
    XGeo = GeoTransform[0]+GeoTransform[1]*Xpixel+Ypixel*GeoTransform[2];
    YGeo = GeoTransform[3]+GeoTransform[4]*Xpixel+Ypixel*GeoTransform[5];
    return YGeo, XGeo

def chazhi(tif_arr1, tif_arr2):

    if tif_arr1.shape == tif_arr2.shape:

        result = tif_arr2 - tif_arr1
    elif len(tif_arr1.shape) > 2:

        flag = tif_arr1.shape - tif_arr2.shape
        if flag[1] > 0:
            pass




if __name__ == "__main__":

    tifPath = r".\clip\GF1_PMS1_E117.1_N28.3_20230918_L1A13090694001-PAN1_rc_rpcortho_clip2_clip.tif"
    outPath = r".\reconstructed\20230918_reconstructed_resample.tif"
    mutilWaveletsChange(tifPath, -1, 4, outPath)


    tifPath = r".\clip\GF1_PMS2_E117.3_N28.2_20230628_L1A0007366742-PAN_rc_rpcortho_clip2_clip.tif"
    outPath = r".\reconstructed\20230628_reconstructed_resample.tif"
    mutilWaveletsChange(tifPath, -1, 4, outPath)

