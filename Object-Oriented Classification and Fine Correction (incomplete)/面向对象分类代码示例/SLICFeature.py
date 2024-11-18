# --coding:utf-8--
from skimage.segmentation import mark_boundaries
from osgeo import gdal, ogr
from skimage.segmentation import slic, quickshift
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ReadWrite_h5 as rw
from ReadShp import ARCVIEW_SHAPE
from sklearn.ensemble import RandomForestClassifier


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
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def get_SLIC_deffirence(img_path1, img_path2):

    temp_path = r".\Atmosphere\SLIC\230128_SLIC.tif"
    im_width, im_height, im_proj, im_geotrans, im_data = read_img(img_path1)
    im_data = im_data[0:3]
    temp = im_data.transpose((2, 1, 0))
    # segments_quick = quickshift(temp, kernel_size=3, max_dist=6, ratio=0.5)
    segments_quick = slic(temp, n_segments=1000, compactness=20)
    print(segments_quick.shape)
    print(im_data.shape)
    mark0 = mark_boundaries(temp, segments_quick)
    re0 = mark0.transpose((2, 1, 0))
    write_img(temp_path, im_proj, im_geotrans, re0)


    temp_path2 = r".\Atmosphere\SLIC\231031_SLIC.tif"
    im_width2, im_height2, im_proj2, im_geotrans2, im_data2 = read_img(img_path2)
    im_data2 = im_data2[0:3]
    temp2 = im_data2.transpose((2, 1, 0))
    # segments_quick = quickshift(temp, kernel_size=3, max_dist=6, ratio=0.5)
    segments_quick2 = slic(temp2, n_segments=1000, compactness=20)
    print(segments_quick2.shape)
    print(im_data2.shape)
    mark02 = mark_boundaries(temp2, segments_quick2)
    re02 = mark02.transpose((2, 1, 0))
    write_img(temp_path2, im_proj2, im_geotrans2, re02)

    write_img(r'.\Atmosphere\SLIC\1031_segments_quick2.tif', im_proj2, im_geotrans2, segments_quick2.transpose((1, 0)))


def get_SLIC_attributeFeature(arr_index, tifArr):

    regions_attributes = regionprops(arr_index.astype(np.int32))
    attributes_dict = {}
    for attribute in regions_attributes:
        attributes_dict[attribute.label] = {
            'area': attribute.area,
            'major_length': attribute.axis_major_length,
            'perimeter': attribute.perimeter,
            'euler_number': attribute.euler_number,
            'extent': attribute.extent,
            'convex_area_percent': attribute.area_convex
        }

    return attributes_dict


def get_image_lightInformation(arr_index, tifArr):

    index_list = list(set(arr_index.reshape(-1).astype(np.int32).tolist()))
    band1 = tifArr[0]
    band2 = tifArr[1]
    band3 = tifArr[2]
    band4 = tifArr[3]
    band_dict = {}
    ndvi = (band4 - band3) / (band4 + band3)
    ndwi = (band2 - band4) / (band2 + band4)
    for index in index_list:

        band_dict[index] = {
            'blue_mean': np.nanmean(band1[arr_index == index]),
            'green_mean': np.nanmean(band2[arr_index == index]),
            'red_mean': np.nanmean(band3[arr_index == index]),
            'nir_mean': np.nanmean(band1[arr_index == index]),

            'blue_std': np.nanstd(band1[arr_index == index]),
            'green_std': np.nanstd(band2[arr_index == index]),
            'red_std': np.nanstd(band3[arr_index == index]),
            'nir_std': np.nanstd(band1[arr_index == index]),

            'blue_max': np.nanmax(band1[arr_index == index]),
            'green_max': np.nanmax(band2[arr_index == index]),
            'red_max': np.nanmax(band3[arr_index == index]),
            'nir_max': np.nanmax(band1[arr_index == index]),

            'blue_min': np.nanmin(band1[arr_index == index]),
            'green_min': np.nanmin(band2[arr_index == index]),
            'red_min': np.nanmin(band3[arr_index == index]),
            'nir_min': np.nanmin(band1[arr_index == index]),

            'ndvi_mean': np.nanmean(ndvi[arr_index == index]),
            'ndvi_std': np.nanstd(ndvi[arr_index == index]),
            'ndvi_max': np.nanmax(ndvi[arr_index == index]),
            'ndvi_min': np.nanmin(ndvi[arr_index == index]),

            'ndwi_mean': np.nanmean(ndwi[arr_index == index]),
            'ndwi_std': np.nanstd(ndwi[arr_index == index]),
            'ndwi_max': np.nanmax(ndwi[arr_index == index]),
            'ndwi_min': np.nanmin(ndwi[arr_index == index]),
        }
    return band_dict


def get_imageGLCM(arr_index, glcm_arr):
    arr_index = arr_index.astype(np.int32)
    glcm_dict = {}
    index_list = list(set(arr_index.reshape(-1).astype(np.int32)))
    for index in index_list:

        glcm_dict[index] ={
            'glcm_mean_meant': np.nanmean(glcm_arr[arr_index == index]),
            'glcm_mean_std': np.nanstd(glcm_arr[arr_index == index]),
            'glcm_mean_max': np.nanmax(glcm_arr[arr_index == index]),
            'glcm_mean_min': np.nanmin(glcm_arr[arr_index == index])
        }

    return glcm_dict


def get_Label(arr_index, coordList, transform):
    arr_index = arr_index.astype(np.int32)
    label_dict = {}
    for i in range(coordList.shape[0]):
        SLICcol, SLICrow = rw.geo2imagexy(coordList[i][0], coordList[i][1], transform)
        SLIC_label = arr_index[int(SLICrow), int(SLICcol)]
        label_dict[SLIC_label] = {
            'classfiterLabel': coordList[i][2]
        }
    print(label_dict)
    return label_dict


def GetTrainData(indexPath, tifPath, glcmPath, csvPath, model='train'):
    indexDataset = gdal.Open(indexPath)
    tifDataset = gdal.Open(tifPath)
    glcmDataset = gdal.Open(glcmPath)

    projection = indexDataset.GetProjection()
    transform = tifDataset.GetGeoTransform()

    imgx, imgy = indexDataset.RasterXSize, indexDataset.RasterYSize
    indexArr = indexDataset.ReadAsArray(0, 0, imgx, imgy)
    tifArr = tifDataset.ReadAsArray(0, 0, imgx, imgy)

    glcm_arr = glcmDataset.ReadAsArray(0, 0, imgx, imgy)

    feature_dict = get_SLIC_attributeFeature(indexArr, tifArr)
    feature_df = pd.DataFrame.from_dict(feature_dict)
    # print(feature_df)

    light_Information = get_image_lightInformation(indexArr, tifArr)
    light_attribute_df = pd.DataFrame.from_dict(light_Information)

    glcm_dict = get_imageGLCM(indexArr, glcm_arr)
    glcm_df = pd.DataFrame.from_dict(glcm_dict)
    # print(light_attribute_df)
    if model == 'train':
    # 获取shp点的经纬度转化为行列号读取相应位置的index
        test = ARCVIEW_SHAPE()
        spatialref, geomtype, geomlist, fieldlist, reclist = test.read_shp(r".\data\TrainFeaturePoint\LabelPoints.shp")
        coordList = []
        for i in range(len(reclist)):
            # print(geomlist[i].split(' '))
            _, coordx, coordy = geomlist[i].split(' ')
            coordx = coordx[1:]
            coordy = coordy[:-2]
            coordList.append(np.array([float(coordx), float(coordy), reclist[i]['Label']]))

        coordList = np.array(coordList)
        # print(coordList)

        label_dict = get_Label(indexArr, coordList, transform)
        label_df = pd.DataFrame.from_dict(label_dict)
        print(label_df)

        dfList = pd.concat([feature_df, light_attribute_df, glcm_df, label_df])
        print(dfList)

        dfList = dfList.dropna(axis=1)
        dfList = dfList.T
        dfList.to_csv(csvPath)

        return dfList.to_numpy()
    else:
        dfList = pd.concat([feature_df, light_attribute_df, glcm_df])
        print(dfList)

        dfList = dfList.dropna(axis=1)
        dfList = dfList.T

        return dfList.index.to_numpy(), dfList.to_numpy()


def Predict(indexPath, tifPath, glcmPath, model, outPath):

    index_list, dataset = GetTrainData(indexPath, tifPath, glcmPath, csvPath, model='predict')
    result = model.predict(dataset)

    indexDataset = gdal.Open(indexPath)
    tifDataset = gdal.Open(tifPath)
    glcmDataset = gdal.Open(glcmPath)

    imgx, imgy = indexDataset.RasterXSize, indexDataset.RasterYSize

    index_arr = indexDataset.ReadAsArray(0, 0, imgx, imgy)
    tif_arr = tifDataset.ReadAsArray(0, 0, imgx, imgy)
    glcm_arr = glcmDataset.ReadAsArray(0, 0 ,imgx, imgy)

    projection, transform = rw.get_GeoInformation(tifDataset)

    predict_arr = np.zeros_like(tif_arr[0])
    for i in range(len(index_list)):
        print(index_arr.shape, predict_arr.shape)
        print(index_list[i], result[i])
        predict_arr[index_arr == index_list[i]] = result[i]

    rw.write_tif(predict_arr, projection, transform, outPath)




if __name__ == "__main__":


    indexPath = r".\data\SLIC\1031_segments_quick2.tif"
    tifPath = r".\data\Resample\231031_resample_clip.tif"
    glcmPath = r'.\data\GLCM\231031_resample_clip_glcmMEANBand4.tif'
    csvPath = r'.\data\TrainData.csv'

    TrainData = GetTrainData(indexPath, tifPath, glcmPath, csvPath)
    trainX, trainY = TrainData[:, :-1], TrainData[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(trainX, trainY, test_size=0.2, random_state=5)
    randomForest = RandomForestClassifier(n_estimators=500, max_depth=15)

    randomForest.fit(train_x, train_y)

    Predict(indexPath, tifPath, glcmPath, randomForest, r'.\data\Classfilter\20231031_result.tif')



