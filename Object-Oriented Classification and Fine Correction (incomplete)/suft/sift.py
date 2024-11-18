# --coding:utf-8--
import glob
import PIL.Image
import numpy as np
import cv2
from osgeo import gdal, osr


def gdalReadImage(inputPath, noData=65536):

    tifDataset = gdal.Open(inputPath, gdal.GA_Update)
    projection = defineSRS(4326)[0].ExportToWkt()

    imgx, imgy = tifDataset.RasterXSize, tifDataset.RasterYSize
    tifArr = tifDataset.ReadAsArray(0, 0, imgx, imgy)
    tifArr[tifArr == 65536] = 0

    return tifDataset, tifArr


def get_GeoInformation(dataset):

    projection = dataset.GetProjection()
    transform = dataset.GetGeoTransform()

    return projection, transform


def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def ReadImage(inputpath):
    img = np.array(PIL.Image.open(inputpath)).astype(np.float32)
    return img


def GetImageFeature(img1, img2):

    # print(img1[100, 100])
    sift = cv2.SIFT_create(5000)

    img1 = normalization(img1)
    img2 = normalization(img2)
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGBA)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2RGBA)
    gray_img1 = Tiff16to8bit(gray_img1)
    gray_img2 = Tiff16to8bit(gray_img2)

    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 或传递一个空字典
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)


    # 只需要绘制好匹配项，因此创建一个掩码
    matchesMask = [[0, 0] for i in range(len(matches))]
    # 根据Lowe的论文进行比例测试


    # for m in good:
        # print(m.queryIdx)
        # print(kp1[m.queryIdx].pt, kp2[m.queryIdx].pt)
    # exit(0)
    ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    print(np.squeeze(ptsA).shape, np.squeeze(ptsB).shape)

    ptsA, ptsB = nuqiue_element(np.squeeze(ptsA), np.squeeze(ptsB))

    # M, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 1)
    # matchesMask = mask.ravel().tolist()
    #
    # ptsA_final = []
    # ptsB_final = []
    # for i in range(len(mask)):
    #     if mask[i]:
    #         ptsA_final.append(ptsA[i])
    #         ptsB_final.append(ptsB[i])

    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.7 * n.distance:
    #         matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(gray_img1, kp1, gray_img2, kp2, matches, None, **draw_params)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)

    return np.squeeze(ptsA), np.squeeze(ptsB)


def GCPCorrect(pstB, pstA_lon, pstA_lat, srcDataset, outputPath):

    GCPS = []

    print(pstA_lat.shape, pstA_lon.shape, 0, pstB.shape)
    for i in range(pstB.shape[0]):
        # print(pstA_lat[i], pstA_lon[i], 0, pstB[i, 0], pstB[i, 1])
        GCP = gdal.GCP(float(pstA_lon[i]), float(pstA_lat[i]), 0, float(pstB[i, 1]), float(pstB[i, 0]))
        # GCP = gdal.GCP(35.470800493273345, 33.87513999833673, 0, 8042, 6818)
        GCPS.append(GCP)


    projection, transform = get_GeoInformation(srcDataset)
    # 添加控制点

    transform = srcDataset.GetGeoTransform()
    x_res = abs(transform[1])
    y_res = abs(transform[5])
    # tps校正 重采样:最邻近法

    tiff_driver = gdal.GetDriverByName('MEM')
    temp_ds = tiff_driver.CreateCopy('', srcDataset, 0)
    print(GCPS[0])
    temp_ds.SetGCPs(GCPS, projection)

    dst_ds = gdal.Warp(outputPath, temp_ds, format='GTiff', tps=True, dstSRS=projection,
                       xRes=x_res, yRes=y_res, resampleAlg=gdal.GRIORA_NearestNeighbour,
                       dstNodata=65535, srcNodata=-1, outputType=gdal.GDT_Int32)


def SURF(img1, img2):
    surf = cv2.xfeatures2d.SURF_create()

    print(surf)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def Tiff16to8bit(img_16):
    if (np.max(img_16) - np.min(img_16) != 0):
        # img_nrm = (img_16 - np.min(img_16)) / (np.max(img_16) - np.min(img_16)) #计算灰度范围,归一化
        img_nrm = normalization(img_16)
        img_8 = np.uint8(255 * img_nrm)
    return img_8


def nuqiue_element(ptsA, ptsB):

    ptsA_unqiue, ptsA_counts = np.unique(ptsA, axis=0, return_counts=True)
    ptsB_unqiue, ptsB_counts = np.unique(ptsB, axis=0, return_counts=True)
    ptsA_repeated_index = np.where(ptsA_counts > 1)[0]
    ptsB_repeated_index = np.where(ptsB_counts > 1)[0]
    all_arg = np.append(ptsA_repeated_index, ptsB_repeated_index)
    unqiue_all_arg = np.unique(all_arg)
    # print(ptsA_repeated_index)
    # ptsA = np.delete(ptsA, unqiue_all_arg[:, np.newaxis], axis=0)
    # ptsB = np.delete(ptsB, unqiue_all_arg[:, np.newaxis], axis=0)
    # exit(0)
    apha_arr = (ptsA[:, 0] - ptsB[:, 0]) / (ptsA[:, 1] - ptsB[:, 1])
    bin_edges = [i for i in range(-360, 361, 1)]
    hist, edges = np.histogram(apha_arr, bins=bin_edges)
    most_hist_index = np.argmax(hist)
    most_common = (edges[most_hist_index], edges[most_hist_index + 1])
    new_ptsA = []
    new_ptsB = []
    for i in range(ptsA.shape[0]):

        if (apha_arr[i] <= edges[most_hist_index + 1]) & (apha_arr[i] <= edges[most_hist_index - 1]):
            new_ptsA.append(ptsA[i])
            new_ptsB.append(ptsB[i])
    new_ptsA = np.array(new_ptsA)
    new_ptsB = np.array(new_ptsB)

    new_ptsA = np.unique(new_ptsA, axis=0)
    new_ptsB = np.unique(new_ptsB, axis=0)

    return new_ptsA, new_ptsB


def defineSRS(reference_num):

    prosrs = osr.SpatialReference()
    prosrs.ImportFromEPSG(reference_num)
    geosrs = prosrs.CloneGeogCS()

    return prosrs, geosrs

if __name__ == '__main__':

    tifPath1 = r"J:\0KTYensemble\000 Chemical\nternalCalibration\20230128_clip.tif"
    tifPath2 = r"J:\0KTYensemble\000 Chemical\nternalCalibration\20231031_clip_resample.tif"

    tifDataset1, img1 = gdalReadImage(tifPath1)

    img1 = np.transpose(img1, (1, 2, 0)).astype(np.float32)

    tifDataset2, img2 = gdalReadImage(tifPath2)
    img2 = np.transpose(img2, (1, 2, 0)).astype(np.float32)

    srcPoint, pwPoint = GetImageFeature(img1, img2)

    srcPointLon, srcPointLat = imagexy2geo(tifDataset1, srcPoint[:, 0], srcPoint[:, 1])
    pwPointLon, pwPointLat = imagexy2geo(tifDataset2, pwPoint[:, 0], pwPoint[:, 1])

    # print(srcPointLon[0], pwPointLon[0])
    # print(srcPointLat[0], pwPointLat[0])
    # print(srcPoint[0], pwPoint[0])
    GCPCorrect(srcPoint, pwPointLon, pwPointLat, tifDataset2, r'J:\0KTYensemble\2020 黎巴嫩布鲁特港口\231212 jjz\231215_test.tif')