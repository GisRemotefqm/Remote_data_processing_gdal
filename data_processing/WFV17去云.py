from osgeo import gdal
import os
from DataMath import ReadWrite_h5
import numpy as np
import cv2

# use image L reflence light
def math_rb_grb(inputpath):

    tif_dataset = ReadWrite_h5.get_tifDataset(inputpath)
    pro, geo = ReadWrite_h5.get_GeoInformation(tif_dataset)
    imgx, imgy = ReadWrite_h5.get_RasterXY(tif_dataset)
    tif_arr = ReadWrite_h5.get_RasterArr(tif_dataset, imgx, imgy)
    red = tif_arr[2, :, :]

    green = tif_arr[1, :, :]

    blue = tif_arr[0, :, :]

    rb = red = blue

    grb = 4 * green - red - 3 * blue

    write_tif(rb, pro, geo, r'D:\测试文件\wfv_rb.tif')
    write_tif(grb, pro, geo, r'D:\测试文件\wfv_grb.tif')

    return rb, grb


def write_tif(data_list, projection, transform, outputPath):
    """
    导出为tif，但是现在没改只能单通道输出
    :param data_list: 整个数据文件
    :param projection: 投影信息，GF-5中妹有投影信息
    :param transform: 仿射变换参数
    :param outputPath: 输出路径
    :return:
    """
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
            # ds.SetProjection(projection.ExportToWkt())
        for i in range(data_list.shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(data_list[i])
        ds.FlushCache()
        del ds
    else:
        row, col = data_list.shape[0], data_list.shape[1]

        driver = gdal.GetDriverByName('GTiff')

        ds = driver.Create(outputPath, col, row, 1, datatype)

        ds.SetGeoTransform(transform)
        ds.SetProjection(projection)
        # ds.SetProjection(projection.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(data_list)
        ds.FlushCache()
        del ds


"""计算灰度共生矩阵"""
def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=11, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)
    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    # digitize
    bins = np.linspace(vmin, vmax + 1, levels + 1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance * np.cos(np.deg2rad(angle))
    dy = distance * np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    h, w = img.shape
    gl2 = cv2.warpAffine(gl1, mat, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


"""计算纹理特征"""
def fast_textural_features(glcm, vmin=0, vmax=255, levels=8, kernel_size=11):
    _, _, h, w = glcm.shape
    # 计算最大值
    max_ = np.max(glcm, axis=(0, 1))
    # 计算熵
    pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / kernel_size ** 2
    ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
    # 均值/标准差/对比度/相异性/同质性/角二阶距
    mean, std2, cont, diss, homo, asm = [np.zeros((h, w), dtype=np.float32) for x in range(6)]
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i, j] * i / (levels) ** 2  # 计算均值
            cont += glcm[i, j] * (i - j) ** 2  # 计算对比度
            diss += glcm[i, j] * np.abs(i - j)  # 计算相异性
            homo += glcm[i, j] / (1. + (i - j) ** 2)  # 计算同质性
            asm += glcm[i, j] ** 2  # 计算角二阶距
    # 计算能量
    ene = np.sqrt(asm)
    # 计算标准差
    for i in range(levels):
        for j in range(levels):
            std2 += (glcm[i, j] * i - mean) ** 2
    std = np.sqrt(std2)

    return [max_, ent, mean, std, cont, diss, homo, asm, ene]


if __name__ == '__main__':

    path = r'D:\20200104\GF1_WFV2_E119.6_N36.0_20200104_L1A0004525722rp.tif'
    dataset = ReadWrite_h5.get_tifDataset(path)
    pro, trans = ReadWrite_h5.get_GeoInformation(dataset)
    imgx, imgy = ReadWrite_h5.get_RasterXY(dataset)
    tif_arr = ReadWrite_h5.get_RasterArr(dataset, imgx, imgy)

    rb, grb = math_rb_grb(path)

    rb_h = np.unique(rb).shape[0]
    grb_h = np.unique(grb).shape[0]
    print(rb_h, grb_h)

    rb_glcm = fast_glcm(rb, vmin=int(np.min(rb)), vmax=int(np.max(rb)) + 1, levels=6, angle=0)
    grb_glcm = fast_glcm(grb, vmin=int(np.min(grb)), vmax=int(np.max(grb)) + 1, levels=6, angle=0)

    rb_mean = fast_textural_features(rb_glcm)[2]
    grb_mean = fast_textural_features(grb_glcm)[2]

    # # get band1 < 0.2 arg is A
    # query_blue = np.argwhere(tif_arr[0, :, :] < 0.2)
    #
    # # get rb and grb  arg is B
    # query_rb = np.argwhere((rb <= 17) & (rb >= 9))
    # query_grb = np.argwhere((grb <= -6) & (rb >= -9.5))
    #
    # # get rb_mean and grb_mean arg is C
    # query_rb_mean = np.argwhere(((rb_mean >= 0) & (rb_mean <= 0.125)) | ((rb_mean >= 0.2) & (rb_mean <= 0.4)))
    # query_grb_mean = np.argwhere(((grb_mean >= 0) & (grb_mean <= 0.125)) | ((grb_mean >= 0.2) & (grb_mean <= 0.4)))

    query_result = np.argwhere((((tif_arr[0, :, :] < 17.5) & (tif_arr[0, :, :] > 4)) & ((rb <= 17) & (rb >= 9) & (grb <= -6) & (grb >= -9.5))) |
                               (((tif_arr[0, :, :] < 17.5) & (tif_arr[0, :, :] > 4)) & (((rb_mean >= 0) & (rb_mean <= 0.125)) | ((rb_mean >= 0.2) & (rb_mean <= 0.4)) &
                                                            (((grb_mean >= 0) & (grb_mean <= 0.125)) | ((grb_mean >= 0.2) & (grb_mean <= 0.4))))))

    for i in range(query_result.shape[0]):
        tif_arr[:, query_result[i, 0], query_result[i, 1]] = 0

    write_tif(tif_arr, pro, trans, r'D:\测试文件\wfv_test_cloud.tif')