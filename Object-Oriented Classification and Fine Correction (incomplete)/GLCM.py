from osgeo import gdal
import os
import numpy as np
import cv2
import glob


# use image L reflence light
def math_rb_grb(tif_arr):

    red = tif_arr[2, :, :]

    green = tif_arr[1, :, :]

    blue = tif_arr[0, :, :]

    rb = red - blue

    grb = 4 * green - red - 3 * blue

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
def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=2, distance=1.0, angle=0.0):
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
def fast_textural_features(glcm, levels=8, kernel_size=2):

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

    paths = glob.glob(r"K:\BMXM\peizhun\clip\*.tif")

    for path in paths:

        dataset = gdal.Open(path)
        pro, trans = dataset.GetProjection(), dataset.GetGeoTransform()
        imgx, imgy = dataset.RasterXSize, dataset.RasterYSize
        tif_arr = dataset.ReadAsArray(0, 0, imgx, imgy)

        band3 = tif_arr[3]

        rb_glcm = fast_glcm(band3, vmin=int(np.min(band3)), vmax=int(np.max(band3)), levels=17, kernel_size=7, angle=0)

        max_, ent, mean, std, cont, diss, homo, asm, ene = fast_textural_features(rb_glcm, levels=17, kernel_size=7)

        write_tif(max_, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'max_.tif'))
        write_tif(ent, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'ent.tif'))
        write_tif(mean, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'mean.tif'))
        write_tif(std, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'std.tif'))
        write_tif(cont, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'cont.tif'))
        write_tif(diss, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'diss.tif'))
        write_tif(homo, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'homo.tif'))
        write_tif(asm, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'asm.tif'))
        write_tif(ene, pro, trans,
                  os.path.join(r'K:\BMXM\peizhun\GLCM', os.path.basename(path)[:-4] + 'ene.tif'))
