# --coding:utf-8--
import cv2
from osgeo import gdal
import numpy as np


def MovmarkRandomField(tifPath, classfiler_num, max_iter, outPath):
    dataset = gdal.Open(tifPath)
    projection = dataset.GetProjection()
    transform = dataset.GetGeoTransform()
    imgx, imgy = dataset.RasterXSize, dataset.RasterYSize
    tifArr = dataset.ReadAsArray(0, 0, imgx, imgy)
    print(tifArr.shape)
    # tifArr = tifArr[np.newaxis, :]

    # tifArr = np.transpose(tifArr, (2, 1, 0))
    # print(tifArr.shape)
    # gray = cv2.cvtColor(tifArr, 0)  # 将图片二值化，彩色图片该方法无法做分割
    # img = gray

    # img_double = np.array(tifArr, dtype=np.float64)
    img_double = tifArr.astype(np.float64)
    label = np.random.randint(1, classfiler_num + 1, size=img_double.shape)

    iter = 0

    f_u = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_d = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)
    f_l = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_r = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(3, 3)
    f_ul = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_ur = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_dl = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3)
    f_dr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(3, 3)

    while iter < max_iter:
        iter = iter + 1
        print(iter)

        label_u = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_u)
        label_d = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_d)
        label_l = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_l)
        label_r = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_r)
        label_ul = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ul)
        label_ur = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ur)
        label_dl = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dl)
        label_dr = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dr)
        m, n = label.shape
        p_c = np.zeros((classfiler_num, m, n))

        for i in range(classfiler_num):
            label_i = (i + 1) * np.ones((m, n))
            u_T = 1 * np.logical_not(label_i - label_u)
            d_T = 1 * np.logical_not(label_i - label_d)
            l_T = 1 * np.logical_not(label_i - label_l)
            r_T = 1 * np.logical_not(label_i - label_r)
            ul_T = 1 * np.logical_not(label_i - label_ul)
            ur_T = 1 * np.logical_not(label_i - label_ur)
            dl_T = 1 * np.logical_not(label_i - label_dl)
            dr_T = 1 * np.logical_not(label_i - label_dr)
            temp = u_T + d_T + l_T + r_T + ul_T + ur_T + dl_T + dr_T

            p_c[i, :] = (1.0 / 8) * temp

        p_c[p_c == 0] = 0.001

        mu = np.zeros((1, classfiler_num))
        sigma = np.zeros((1, classfiler_num))
        for i in range(classfiler_num):
            index = np.where(label == (i + 1))
            # data_c = img[index]
            data_c = tifArr[index]
            mu[0, i] = np.mean(data_c)
            sigma[0, i] = np.var(data_c)

        p_sc = np.zeros((classfiler_num, m, n))
        one_a = np.ones((m, n))

        for j in range(classfiler_num):
            MU = mu[0, j] * one_a
            p_sc[j, :] = (1.0 / np.sqrt(2 * np.pi * sigma[0, j])) * np.exp(-1. * ((tifArr - MU) ** 2) / (2 * sigma[0, j]))
            # p_sc[j, :] = (1.0 / np.sqrt(2 * np.pi * sigma[0, j])) * np.exp(-1. * ((img - MU) ** 2) / (2 * sigma[0, j]))
        X_out = np.log(p_c) + np.log(p_sc)
        label_c = X_out.reshape(2, m * n)
        label_c_t = label_c.T
        label_m = np.argmax(label_c_t, axis=1)
        label_m = label_m + np.ones(label_m.shape)  # 由于上一步返回的是index下标，与label其实就差1，因此加上一个ones矩阵即可
        label = label_m.reshape(m, n)

    label = label - np.ones(label.shape)  # 为了出现0
    label = label.T
    write_tif(label, projection, transform, outputPath=outPath)


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


if __name__ == '__main__':

    tifPath = r"J:\发送文件\Atmosphere\SLIC\230128_erfenlei_clip_class.tif"
    outPath = r"J:\发送文件\Atmosphere\SLIC\230128_erfenlei_clip_class_markov.tif"
    MovmarkRandomField(tifPath, 2, 200, outPath)

    # tifPath = r"J:\0KTYensemble\000 Chemical\nternalCalibration\20231031_reconstructed_resample.tif"
    # outPath = r"J:\0KTYensemble\000 Chemical\nternalCalibration\20231031_MovRandomField.tif"
    # MovmarkRandomField(tifPath, 2, 200, outPath)