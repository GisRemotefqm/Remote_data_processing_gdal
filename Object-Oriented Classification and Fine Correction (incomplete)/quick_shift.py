# --coding:utf-8--
from skimage.segmentation import mark_boundaries
from osgeo import gdal, ogr, osr
from skimage.segmentation import slic, quickshift
from skimage.measure import regionprops
import numpy as np
import pandas as pd


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


def get_SLIC_deffirence(img_path1, img_path2):
    for kernel_size in range(3, 18, 2):
        for max_dist in range(1, 11):
            for radio in range(1, 10):
                temp_path = r"J:\发送文件\Atmosphere\quickshift\230128_rpc_clip_quickShift_kernelSize=" + str(kernel_size) + "_maxDist=" + str(max_dist) + "radio=" + str(radio * 0.1) + ".tif"
                im_width, im_height, im_proj, im_geotrans, im_data = read_img(img_path1)
                im_data = im_data[:3]
                temp = im_data.transpose((2, 1, 0))
                print(temp.shape)
                segments_quick = quickshift(temp, kernel_size=kernel_size, max_dist=max_dist, ratio=radio * 0.1)

                mark0 = mark_boundaries(temp, segments_quick)
                re0 = mark0.transpose((2, 1, 0))
                print(re0.shape)
                write_img(temp_path, im_proj, im_geotrans, re0)
                # write_img(r'J:\发送文件\Atmosphere\SLIC\230128_rpc_clip_segments_quick2_segments=' + str(segments) + '_compact=' + str(compact) + ".tif",
                #           im_proj, im_geotrans, segments_quick.transpose((1, 0)))

                temp_path2 = r"J:\发送文件\Atmosphere\quickshift\231031_rpc_clip_quickShift_rpc_clip_quickShift_kernelSize=" + str(kernel_size) + "_maxDist=" + str(max_dist) + "radio=" + str(radio * 0.1) + ".tif"
                im_width2, im_height2, im_proj2, im_geotrans2, im_data2 = read_img(img_path2)
                im_data2 = im_data2[:3]
                temp2 = im_data2.transpose((2, 1, 0))
                segments_quick2 = quickshift(temp2, kernel_size=kernel_size, max_dist=max_dist, ratio=radio * 0.1)

                mark02 = mark_boundaries(temp2, segments_quick2)
                re02 = mark02.transpose((2, 1, 0))
                write_img(temp_path2, im_proj2, im_geotrans2, re02)

            # write_img(r'J:\发送文件\Atmosphere\SLIC\231031_rpc_clip_segments_quick2_segments=' + str(segments) + '_compact=' + str(compact) + ".tif",
            #           im_proj2, im_geotrans2, segments_quick2.transpose((1, 0)))


"""
def Raster2Shp():
    train_fn = r".\train_data1.shp"
    train_ds = ogr.Open(train_fn)
    lyr = train_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', im_width, im_height, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(im_geotrans)
    target_ds.SetProjection(im_proj)
    options = ['ATTRIBUTE=tyPE']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    data = target_ds.GetRasterBand(1).ReadAsArray()
    ground_truth = target_ds.GetRasterBand(1).ReadAsArray()
    ground_truth = ground_truth.transpose((1, 0))
    classes = np.unique(ground_truth)[1:]
def Raster2Shp():
    train_fn = r".\train_data1.shp"
    train_ds = ogr.Open(train_fn)
    lyr = train_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', im_width, im_height, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(im_geotrans)
    target_ds.SetProjection(im_proj)
    options = ['ATTRIBUTE=tyPE']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    data = target_ds.GetRasterBand(1).ReadAsArray()
    ground_truth = target_ds.GetRasterBand(1).ReadAsArray()
    ground_truth = ground_truth.transpose((1, 0))
    classes = np.unique(ground_truth)[1:]


"""


def PolygonizeTheRaster(inputfile, outputfile):
    dataset = gdal.Open(inputfile, gdal.GA_ReadOnly)
    srcband = dataset.GetRasterBand(1)
    im_proj = dataset.GetProjection()
    prj = osr.SpatialReference()
    prj.ImportFromWkt(im_proj)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = drv.CreateDataSource(outputfile)
    dst_layername = 'out'
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=prj)
    dst_fieldname = 'DN'
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0
    gdal.Polygonize(srcband, None, dst_layer, dst_field)


if __name__ == "__main__":

    img_path1 = r"J:\发送文件\Atmosphere\jjj\Result\Resample\230128_resample_clip.tif"
    img_path2 = r"J:\发送文件\Atmosphere\jjj\Result\Resample\231031_resample_clip.tif"

    get_SLIC_deffirence(img_path1, img_path2)
