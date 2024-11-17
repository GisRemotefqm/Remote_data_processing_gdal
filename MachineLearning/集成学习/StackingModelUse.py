import os
import numpy as np
from osgeo import gdal
import joblib
from DataMath import ReadWrite_h5
from GetData import GetImageData
import glob
import time


def machine_tif(models, second_model, tif_name, blh_name, r_name, sp_name, wd_name, ws_name, t2m_name, demPath, popPath, outputPath):
    # joblib_file = machine_file
    # forest_model = joblib.load(joblib_file)
    start_time = time.time()
    date = os.path.basename(tif_name).split('_')[0]
    print('方法里', date)

    # 这里是读取气象、TOA、人口、DEM数据对象
    tif = gdal.Open(tif_name)
    blh = gdal.Open(blh_name)
    r = gdal.Open(r_name)
    sp = gdal.Open(sp_name)
    wd = gdal.Open(wd_name)
    ws = gdal.Open(ws_name)
    t2m = gdal.Open(t2m_name)
    dem = gdal.Open(demPath)
    pop = gdal.Open(popPath)

    r_transform = ReadWrite_h5.get_GeoInformation(r)[1]
    blh_transform = ReadWrite_h5.get_GeoInformation(blh)[1]
    sp_transform = ReadWrite_h5.get_GeoInformation(sp)[1]
    t2m_transform = ReadWrite_h5.get_GeoInformation(t2m)[1]
    ws_transform = ReadWrite_h5.get_GeoInformation(ws)[1]
    wd_transform = ReadWrite_h5.get_GeoInformation(wd)[1]
    dem_transform = ReadWrite_h5.get_GeoInformation(dem)[1]
    pop_transform = ReadWrite_h5.get_GeoInformation(pop)[1]

    # 读取影像transform(影像是有偏移和旋转的)，那就直接读取他的左上、右下经纬度，这样就可以把整个数据矩阵框选出来
    tif_x, tif_y = ReadWrite_h5.get_RasterXY(tif)
    tif_arr = ReadWrite_h5.get_RasterArr(tif, tif_x, tif_y)
    print(tif_arr.shape, 'this is tif_arr.shape')
    tif_arr[tif_arr == -1] = np.nan

    ndvi = (tif_arr[3, :, :] - tif_arr[2, :, :]) / (tif_arr[3, :, :] + tif_arr[2, :, :])
    # ndvi = np.nan_to_num(ndvi, nan=0)
    tif_arr = tif_arr[:2, :, :]
    tif_projection, tif_transform = ReadWrite_h5.get_GeoInformation(tif)

    # 左上，通过仿射变换参数获得经纬度
    lt_lon, lt_lat = ReadWrite_h5.imagexy2geo(tif, 0, 0)

    rb_lon, rb_lat = ReadWrite_h5.imagexy2geo(tif, tif_x, tif_y)

    # 以TOA数据的经纬度为准，获得气象数据等数据对应范围的数据矩阵行列号
    r_lt_x, r_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, r_transform)
    print(r_lt_x, r_lt_y)
    r_lt_x, r_lt_y = int(r_lt_x), int(r_lt_y)
    r_rb_x, r_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, r_transform)
    print(r_rb_x, r_rb_y)
    r_rb_x, r_rb_y = int(r_rb_x) - r_lt_x, int(r_rb_y) - r_lt_y

    print(r_lt_y, r_lt_x, r_rb_x, r_rb_y)

    sp_lt_x, sp_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, sp_transform)
    sp_lt_x, sp_lt_y = int(sp_lt_x), int(sp_lt_y)
    sp_rb_x, sp_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, sp_transform)
    sp_rb_x, sp_rb_y = int(sp_rb_x) - sp_lt_x, int(sp_rb_y) - sp_lt_y

    wd_lt_x, wd_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, wd_transform)
    wd_lt_x, wd_lt_y = int(wd_lt_x), int(wd_lt_y)
    wd_rb_x, wd_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, wd_transform)
    wd_rb_x, wd_rb_y = int(wd_rb_x) - wd_lt_x, int(wd_rb_y) - wd_lt_y

    ws_lt_x, ws_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, ws_transform)
    ws_lt_x, ws_lt_y = int(ws_lt_x), int(ws_lt_y)
    ws_rb_x, ws_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, ws_transform)
    ws_rb_x, ws_rb_y = int(ws_rb_x) - ws_lt_x, int(ws_rb_y) - ws_lt_y

    blh_lt_x, blh_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, blh_transform)
    print(blh_lt_x, blh_lt_y)
    blh_lt_x, blh_lt_y = int(blh_lt_x), int(blh_lt_y)
    blh_rb_x, blh_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, blh_transform)
    print(blh_rb_x, blh_rb_y)
    blh_rb_x, blh_rb_y = int(blh_rb_x) - blh_lt_x, int(blh_rb_y) - blh_lt_y

    t2m_lt_x, t2m_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, t2m_transform)
    t2m_lt_x, t2m_lt_y = int(t2m_lt_x), int(t2m_lt_y)
    t2m_rb_x, t2m_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, t2m_transform)
    t2m_rb_x, t2m_rb_y = int(t2m_rb_x) - t2m_lt_x, int(t2m_rb_y) - t2m_lt_y

    dem_lt_x, dem_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, dem_transform)
    dem_lt_x, dem_lt_y = int(dem_lt_x), int(dem_lt_y)
    dem_rb_x, dem_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, dem_transform)
    dem_rb_x, dem_rb_y = int(dem_rb_x) - dem_lt_x, int(dem_rb_y) - dem_lt_y

    pop_lt_x, pop_lt_y = ReadWrite_h5.geo2imagexy(lt_lon, lt_lat, pop_transform)
    pop_lt_x, pop_lt_y = int(pop_lt_x), int(pop_lt_y)
    pop_rb_x, pop_rb_y = ReadWrite_h5.geo2imagexy(rb_lon, rb_lat, pop_transform)
    pop_rb_x, pop_rb_y = int(pop_rb_x) - pop_lt_x, int(pop_rb_y) - pop_lt_y


    # 这里读取到了对应位置的矩阵了，训练时是(12，1)12个特征数据，对应一个pm2.5数据，但是现在都是二维矩阵现在的形状是（12, x, y , 1）对吧
    # 那我就需要转为(12, x * y, 1)这样就有了每行都是13个数据，其中12个是特征数据， 1就是要得到的PM2.5数据
    r_arr = ReadWrite_h5.get_RasterArr(r, lt_x=int(r_lt_x), lt_y=int(r_lt_y), rt_x=int(r_rb_y), rt_y=int(r_rb_x))
    sp_arr = ReadWrite_h5.get_RasterArr(sp, lt_x=int(sp_lt_x), lt_y=int(sp_lt_y), rt_x=int(sp_rb_y), rt_y=int(sp_rb_x))
    wd_arr = ReadWrite_h5.get_RasterArr(wd, lt_x=int(wd_lt_x), lt_y=int(wd_lt_y), rt_x=int(wd_rb_y), rt_y=int(wd_rb_x))
    ws_arr = ReadWrite_h5.get_RasterArr(ws, lt_x=int(ws_lt_x), lt_y=int(ws_lt_y), rt_x=int(ws_rb_y), rt_y=int(ws_rb_x))
    t2m_arr = ReadWrite_h5.get_RasterArr(t2m, lt_x=int(t2m_lt_x), lt_y=int(t2m_lt_y), rt_x=int(t2m_rb_y), rt_y=int(t2m_rb_x))
    blh_arr = ReadWrite_h5.get_RasterArr(blh, lt_x=int(blh_lt_x), lt_y=int(blh_lt_y), rt_x=int(blh_rb_y), rt_y=int(blh_rb_x))
    dem_arr = ReadWrite_h5.get_RasterArr(dem, lt_x=int(dem_lt_x), lt_y=int(dem_lt_y), rt_x=int(dem_rb_y), rt_y=int(dem_rb_x))
    pop_arr = ReadWrite_h5.get_RasterArr(pop, lt_x=int(pop_lt_x), lt_y=int(pop_lt_y), rt_x=int(pop_rb_y), rt_y=int(pop_rb_x))
    # pop_arr = np.log(pop_arr + 1)

    print(pop_arr.shape)
    r_arr = np.nan_to_num(r_arr)
    sp_arr = np.nan_to_num(sp_arr)
    wd_arr = np.nan_to_num(wd_arr)
    ws_arr = np.nan_to_num(ws_arr)
    t2m_arr = np.nan_to_num(t2m_arr)
    blh_arr = np.nan_to_num(blh_arr)
    dem_arr = np.nan_to_num(dem_arr)
    pop_arr = np.nan_to_num(pop_arr)


    # 在这里全部reshape成1维
    tif_arr1 = np.reshape(tif_arr[0], -1)
    tif_arr2 = np.reshape(tif_arr[1], -1)
    blh_arr = np.reshape(blh_arr, -1)
    r_arr = np.reshape(r_arr, -1)
    sp_arr = np.reshape(sp_arr, -1)
    t2m_arr = np.reshape(t2m_arr, -1)
    wd_arr = np.reshape(wd_arr, -1)
    ws_arr = np.reshape(ws_arr, -1)
    dem_arr = np.reshape(dem_arr, -1)
    pop_arr = np.reshape(pop_arr, -1)
    ndvi_3d = np.reshape(ndvi, -1)
    print(tif_arr1.shape, 'this is tif_arr')
    print(tif_arr2.shape, 'this is tif_arr2')
    print(blh_arr.shape, 'this is blh')
    print(r_arr.shape, 'this is r')
    print(sp_arr.shape, 'this is sp')
    print(t2m_arr.shape, 'this is t2m')
    print(wd_arr.shape, 'this is wd')
    print(ws_arr.shape, 'this is ws')
    print(dem_arr.shape, 'this is dem')
    print(pop_arr.shape, 'this is pop')
    print(ndvi.shape, 'this is ndvi')

    # 拼成和训练时一样的形状
    x_test = np.vstack((blh_arr, r_arr, sp_arr, t2m_arr, wd_arr, ws_arr, tif_arr1, tif_arr2, dem_arr, pop_arr, ndvi_3d)).T.astype(np.float32)
    """
    print(x_test.shape)
    x_test = x_test.T
    """

    # 第一层三个网络的输出，预测出的pm2.5数据
    output_img1 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img2 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img3 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img4 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img5 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    print(output_img1.shape, 'this is output_img1.shape')
    output_img1 = output_img1.reshape(-1)
    output_img2 = output_img2.reshape(-1)
    output_img3 = output_img3.reshape(-1)
    output_img4 = output_img4.reshape(-1)
    output_img5 = output_img5.reshape(-1)

    output_imgs = {
        '1': output_img1,
        '2': output_img2,
        '3': output_img3,
        '4': output_img4,
        'second': output_img5
    }
    print(x_test.shape)

    value_test = x_test[:, 0][(x_test[:, 6] > 0) & (~np.isnan(x_test[:, 6])) & (~np.isinf(x_test[:, 6]))]
    print(value_test.shape)
    for i in range(1, x_test.shape[1]):
        value_test = np.vstack(
            (value_test, x_test[:, i][(x_test[:, 6] > 0) &(~np.isnan(x_test[:, 6])) & (~np.isinf(x_test[:, 6]))]))

    value_test = value_test.T.astype(np.float64)

    print(value_test.shape, '预测个数')

    batch_size = 1000000  # 每个批次的大小
    total_samples = value_test.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size  # 计算总共需要多少个批次

    predict_array1 = []
    predict_array2 = []  # 保存预测结果的列表
    predict_array3 = []
    predict_array4 = []
    sencod_result = []
    # 遍历每个批次，将其输入到模型中进行预测，并保存预测结果
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        data_batch = value_test[start_idx:end_idx]

        # 调用模型的预测函数（这里用predict代替），得到批次的预测结果
        batch_predictions1 = models[0].predict(data_batch)
        predict_array1.append(batch_predictions1)

        batch_predictions2 = models[1].predict(data_batch)
        predict_array2.append(batch_predictions2)

        batch_predictions3 = models[2].predict(data_batch)
        predict_array3.append(batch_predictions3)

        # batch_predictions4 = models[3].predict(data_batch)
        # predict_array4.append(batch_predictions4)

        frist_predict = np.array([batch_predictions1, batch_predictions2, batch_predictions3])
        sencond_predictions = second_model.predict(frist_predict.reshape(frist_predict.shape[0], frist_predict.shape[1]).T)
        sencod_result.append(sencond_predictions)
    if predict_array1 == []:
        return

    # 将预测结果合并为一个数组
    predict_array1 = np.concatenate(predict_array1)
    output_imgs[str(1)][(x_test[:, 6] > 0) & (~np.isnan(x_test[:, 6])) & (~np.isinf(x_test[:, 6]))] = predict_array1
    output_imgs[str(1)][x_test[:, 6] < 0] = 0
    output_imgs[str(1)] = output_imgs[str(1)].reshape(tif_y, tif_x)

    predict_array2 = np.concatenate(predict_array2)
    output_imgs[str(2)][(x_test[:, 6] > 0) & (~np.isnan(x_test[:, 6])) & (~np.isinf(x_test[:, 6]))] = predict_array2
    output_imgs[str(2)][x_test[:, 6] < 0] = 0
    output_imgs[str(2)] = output_imgs[str(2)].reshape(tif_y, tif_x)

    predict_array3 = np.concatenate(predict_array3)
    output_imgs[str(3)][(x_test[:, 6] > 0) & (~np.isnan(x_test[:, 6])) & (~np.isinf(x_test[:, 6]))] = predict_array3
    output_imgs[str(3)] = output_imgs[str(3)].reshape(tif_y, tif_x)

    # predict_array4 = np.concatenate(predict_array4)
    # output_imgs[str(4)][(x_test[:, 6] > 0) & (~np.isnan(x_test[:, 6])) & (~np.isinf(x_test[:, 6]))] = predict_array4
    # output_imgs[str(4)] = output_imgs[str(4)].reshape(tif_y, tif_x)

    output_imgs['1'][output_imgs['1'] < 0] = 0
    ReadWrite_h5.write_tif(output_imgs['1'], tif_projection, tif_transform,
                           os.path.join(outputPath, r'ET\\' + date + '.tif'))

    output_imgs['2'][output_imgs['2'] < 0] = 0
    ReadWrite_h5.write_tif(output_imgs['2'], tif_projection, tif_transform,
                           os.path.join(outputPath, r'XGBoost\\' + date + '.tif'))

    output_imgs['3'][output_imgs['3'] < 0] = 0
    ReadWrite_h5.write_tif(output_imgs['3'], tif_projection, tif_transform,
                           os.path.join(outputPath, r'RF\\' + date + '.tif'))

    # output_imgs['4'][output_imgs['4'] < 0] = 0
    # ReadWrite_h5.write_tif(output_imgs['4'], tif_projection, tif_transform,
    #                        os.path.join(outputPath, r'Light\\' + date + '.tif'))"""

    sencod_result = np.concatenate(sencod_result)
    output_imgs['second'][(x_test[:, 6] > 0) & (~np.isnan(x_test[:, 6])) & (~np.isinf(x_test[:, 6]))] = sencod_result
    output_imgs['second'][x_test[:, 6] < 0] = 0
    output_imgs['second'] = output_imgs['second'].reshape(tif_y, tif_x)

    output_imgs['second'][output_imgs['second'] < 0] = 0
    ReadWrite_h5.write_tif(output_imgs['second'], tif_projection, tif_transform,
                           os.path.join(outputPath, r'stacking\\' + date + '.tif'))



    print(tif_arr.shape)
    end_time = time.time()
    run_time = end_time - start_time
    print('运行一次时间为', run_time)


def mat_lonlat(im_geotrans, width, height):
    left_lon = im_geotrans[0]
    top_lat = im_geotrans[3]
    start_lon = left_lon + 0.0005
    end_lon = left_lon + (width - 1) * 0.0009 + 0.0005
    start_lat = top_lat - 0.0005
    end_lat = top_lat - (height - 1) * 0.0009 - 0.0005

    lon_arr = np.linspace(start_lon, end_lon, num=width)
    lat_arr = np.linspace(start_lat, end_lat, num=height)

    lon_mat = np.tile(lon_arr, (height, 1)).astype('float32')  # 生成二维经度矩阵[im_height, im_width]
    lat_mat = lat_arr.repeat(width)
    lat_mat = np.reshape(lat_mat, (height, width)).astype('float32')  # 生成二维纬度矩阵[im_height, im_width]
    print(lon_mat, lat_mat)
    return lon_mat, lat_mat


def machine_AOD(models, tif_name, tcwv_name, tco_name, demPath, outputPath):
    x_test, tif_x, tif_y, tif_projection, tif_transform = GetImageData.GetAODImgData(tif_name, tcwv_name, tco_name, demPath)
    # 第一层三个网络的输出，预测出的pm2.5数据
    output_img1 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img2 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan

    print(output_img1.shape, 'this is output_img1.shape')
    output_imgs = {
        '1': output_img1,
        '2': output_img2,

    }
    print(x_test.shape)

    for i in tqdm(range(0, tif_y)):
        x_array = x_test[i, :, :]
        z_index = np.where(np.isnan(x_array))
        zero_index = np.unique(z_index[0])  # 提取出含有nan的列的索引
        if zero_index.shape[0] < tif_x - 1:
            all_index = np.arange(0, tif_x)
            non_zero_index = np.setdiff1d(all_index, zero_index)  # 去掉有nan的列索引, 保留其余索引
            input_array = x_array[non_zero_index]
            predict_array1 = models[0].predict(input_array)
            output_imgs[str(1)][i, non_zero_index] = predict_array1

            predict_array2 = models[1].predict(input_array)
            output_imgs[str(2)][i, non_zero_index] = predict_array2

    output_imgs['1'][output_imgs['1'] < 0] = 0

    ReadWrite_h5.write_tif(output_imgs['1'], tif_projection, tif_transform,
                           os.path.join(outputPath, r'Light\\' + date + '.tif'))

    output_imgs['2'][output_imgs['2'] < 0] = 0
    ReadWrite_h5.write_tif(output_imgs['2'], tif_projection, tif_transform,
                           os.path.join(outputPath, r'RF\\' + date + '.tif'))


def machine_AOD2(models, tif_name, tcwv_name, tco_name, demPath, outputPath):
    start_time = time.time()
    if os.path.exists(os.path.join(outputPath, date + '.tif')):
        return
    x_test, tif_x, tif_y, tif_projection, tif_transform = GetImageData.GetAOdImgData2(tif_name, tcwv_name, tco_name, demPath)
    print('this is x_test.shape', x_test[:, 6].shape)
    # 第一层三个网络的输出，预测出的pm2.5数据
    output_img1 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img2 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img3 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img4 = np.zeros([tif_y, tif_x], dtype='float32') + np.nan
    output_img1 = output_img1.reshape(-1)
    output_img2 = output_img2.reshape(-1)
    output_img3 = output_img3.reshape(-1)
    output_img4 = output_img4.reshape(-1)
    print(output_img1.shape, 'this is output_img1.shape')
    output_imgs = {
        '1': output_img1,
        '2': output_img2,
        '3': output_img3,
        'second': output_img4
    }
    second_model = models[-1]
    value_test = x_test[:, 0][(x_test[:, 0] > 0) & (~np.isnan(x_test[:, 0])) & (~np.isinf(x_test[:, 0]))]
    print(np.all(x_test))
    for i in range(1, x_test.shape[1]):
        value_test = np.vstack((value_test, x_test[:, i][(x_test[:, 0] > 0) & (~np.isnan(x_test[:, 0])) & (~np.isinf(x_test[:, 0]))]))

    value_test = value_test.T.astype(np.float64)
    print(np.all(value_test))
    value_test = np.nan_to_num(value_test)
    print(value_test.shape, '预测个数')

    batch_size = 1000000  # 每个批次的大小
    total_samples = value_test.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size  # 计算总共需要多少个批次

    predict_array1 = []
    predict_array2 = []  # 保存预测结果的列表
    predict_array3 = []
    sencod_result = []
    # 遍历每个批次，将其输入到模型中进行预测，并保存预测结果
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        data_batch = value_test[start_idx:end_idx]

        # 调用模型的预测函数（这里用predict代替），得到批次的预测结果
        batch_predictions1 = models[0].predict(data_batch)
        predict_array1.append(batch_predictions1)
        batch_predictions2 = models[1].predict(data_batch)
        predict_array2.append(batch_predictions2)
        # batch_predictions3 = models[2].predict(data_batch)
        # predict_array3.append(batch_predictions3)

        frist_predict = np.array([batch_predictions1, batch_predictions2])
        sencond_predictions = second_model.predict(
            frist_predict.reshape(frist_predict.shape[0], frist_predict.shape[1]).T)
        sencod_result.append(sencond_predictions)

    if predict_array1 == []:
        return

    # 将预测结果合并为一个数组
    predict_array1 = np.concatenate(predict_array1)
    output_imgs[str(1)][(x_test[:, 0] > 0) & (~np.isnan(x_test[:, 0])) & (~np.isinf(x_test[:, 0]))] = predict_array1
    output_imgs[str(1)][x_test[:, 6] < 0] = -1
    output_imgs[str(1)] = output_imgs[str(1)].reshape(tif_y, tif_x)
    predict_array2 = np.concatenate(predict_array2)
    output_imgs[str(2)][(x_test[:, 0] > 0) & (~np.isnan(x_test[:, 0])) & (~np.isinf(x_test[:, 0]))] = predict_array2
    output_imgs[str(2)][x_test[:, 6] < 0] = -1
    output_imgs[str(2)] = output_imgs[str(2)].reshape(tif_y, tif_x)
    # predict_array2 = np.concatenate(predict_array2)
    # output_imgs[str(2)][(x_test[:, 0] > 0) & (~np.isnan(x_test[:, 0])) & (~np.isinf(x_test[:, 0]))] = predict_array2
    # output_imgs[str(2)][x_test[:, 6] < 0] = 0
    # output_imgs[str(2)] = output_imgs[str(2)].reshape(tif_y, tif_x)

    # predict_array3 = np.concatenate(predict_array3)
    # output_imgs[str(3)][(x_test[:, 0] > 0) & (~np.isnan(x_test[:, 0])) & (~np.isinf(x_test[:, 0])) & (~np.isnan(x_test[:, 7]))] = predict_array3
    # output_imgs[str(3)] = output_imgs[str(3)].reshape(tif_y, tif_x)

    # output_imgs['1'][output_imgs['1'] < 0] = 0
    # ReadWrite_h5.write_tif(output_imgs['1'], tif_projection, tif_transform,
    #                        os.path.join(outputPath, r'Light\\' + date + '.tif'))

    # output_imgs['2'][output_imgs['2'] < 0] = 0
    # ReadWrite_h5.write_tif(output_imgs['2'], tif_projection, tif_transform,
    #                       os.path.join(outputPath, r'XGB\\' + date + '.tif'))

    # output_imgs['3'][output_imgs['3'] < 0] = 0
    # ReadWrite_h5.write_tif(output_imgs['3'], tif_projection, tif_transform,
    #                        os.path.join(outputPath, r'ET\\' + date + '.tif'))

    sencod_result = np.concatenate(sencod_result)
    output_imgs['second'][(x_test[:, 0] > 0) & (~np.isnan(x_test[:, 0])) & (~np.isinf(x_test[:, 0]))] = sencod_result
    output_imgs['second'][x_test[:, 6] < 0] = -1
    output_imgs['second'] = output_imgs['second'].reshape(tif_y, tif_x)

    output_imgs['second'][output_imgs['second'] < 0] = -1
    ReadWrite_h5.write_tif(output_imgs['second'], tif_projection, tif_transform,
                           os.path.join(outputPath, date + '.tif'))


    end_time = time.time()
    run_time = end_time - start_time
    print('运行一次时间为', run_time)


def get_data(lat_arr, lon_arr, tiff_path, nan_value):
    """
    :param lat_arr: H行像素的纬度, (Height, Width)
    :param lon_arr: H行像素的经度, (Height, Width)
    :param tiff_path: 待提取的tiff图像
    :param nan_value: 等于 nan_value, 判定为nan
    :return: 提取到的5行像素的值
    """
    # data_1d为 Height*Width, 是二维数据(Height, Width)的展平
    data_1d = np.zeros(lat_arr.shape[0]*lon_arr.shape[1]) + np.nan

    lat_1d = lat_arr.flatten()
    lon_1d = lon_arr.flatten()

    dataset = gdal.Open(tiff_path)
    columns = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, columns, rows).astype('float32')
    data[np.where(data <= nan_value)] = np.nan
    geotransform = dataset.GetGeoTransform()

    start_lon = geotransform[0]
    pixel_width = geotransform[1]
    start_lat = geotransform[3]
    pixel_height = geotransform[5] * -1
    # 得到 Height*Width 个像素点所在的 行序号 和 列序号
    row_idx = np.floor((start_lat - lat_1d) / pixel_height).astype('int32')
    col_idx = np.floor((lon_1d - start_lon) / pixel_width).astype('int32')
    # 得到 行序号 和 列序号 同时有效的序号
    row_valid = (row_idx >= 0) & (row_idx <= rows - 1)
    col_valid = (col_idx >= 0) & (col_idx <= columns - 1)
    site_valid = row_valid & col_valid
    # 提取出data中的 有效行列号 对应的数据
    data_1d[site_valid] = data[row_idx[site_valid], col_idx[site_valid]]

    data_2d = np.reshape(data_1d, (lat_arr.shape[0], lon_arr.shape[1])).astype('float32')

    return data_2d


if __name__ == '__main__':

    # firest_XGB = r"K:\231021 武汉成都\武汉MachineLearning\XGBoost\GFALL_XGB_PM10_231115_0.m"
    # firest_ET = r"K:\231021 武汉成都\武汉MachineLearning\ET\GFALL_ET_PM10_231115_0.m"
    # # firest_Light = r"J:\毕业论文\MachineLearning\Light GBM\GF_Light_PM10_231010.m"
    # firest_RF = r"K:\231021 武汉成都\武汉MachineLearning\RF\GFALL_RF_PM10_231115_0.m"
    #
    # RF = joblib.load(firest_RF)
    # XGB = joblib.load(firest_XGB)
    # # Light = joblib.load(firest_Light)
    # ET = joblib.load(firest_ET)
    #
    #
    # # models = [ET, Light, RF]
    # models = [ET, XGB, RF]
    # second_model = joblib.load(r"K:\231021 武汉成都\武汉MachineLearning\Stacking\GFALL_EX_ridge_PM10_231115_0.m")
    # second_model = joblib.load('./去除尼泊尔高值模型/尼泊尔/模型结果/第一层两个模型_第二层模型结果/EX_线性_month_mean_尼泊尔国内_new1.m')

    # firest_RF = r'G:\KongTianYuan\DataOperations\XXX项目\MachineLearning\集成学习\去除尼泊尔高值模型\第一层四个模型结果\firest_RF_month_mean.m'
    # firest_XGB = r'G:\KongTianYuan\DataOperations\XXX项目\MachineLearning\集成学习\去除尼泊尔高值模型\第一层四个模型结果\firest_XGB_month_mean.m'
    # firest_Light = r'G:\KongTianYuan\DataOperations\XXX项目\MachineLearning\集成学习\去除尼泊尔高值模型\第一层四个模型结果\firest_Light_month_mean.m'
    # firest_ET = r'G:\KongTianYuan\DataOperations\XXX项目\MachineLearning\集成学习\去除尼泊尔高值模型\第一层四个模型结果\firest_ET_month_mean.m'

    # tif_path = r'J:\My_PM2.5Data\WuHai\clipTOA\again'
    # blh_path = r'J:\My_PM2.5Data\WuHai\气象数据\WuHai\blh'
    # r_path = r'J:\My_PM2.5Data\WuHai\气象数据\WuHai\r'
    # sp_path = r'J:\My_PM2.5Data\WuHai\气象数据\WuHai\sp'
    # # tp_path = r'J:\K盘气象数据\气象月平均\尼泊尔\tp'
    # wd_path = r'J:\My_PM2.5Data\WuHai\气象数据\WuHai\wd'
    # ws_path = r'J:\My_PM2.5Data\WuHai\气象数据\WuHai\ws'
    # t2m_path = r'J:\My_PM2.5Data\WuHai\气象数据\WuHai\t2m'
    # demPath = r"J:\My_PM2.5Data\WuHai\气象数据\WuHai\other\GMTED2010_Wuhai_resample.tiff"
    # popPath = r"J:\My_PM2.5Data\WuHai\气象数据\WuHai\other\landscan-global-2022_Wuhai_resample.tiff"
    # output = r'J:\My_PM2.5Data\WuHai\Result'
    # for temp in os.listdir(tif_path):
    #
    #     if os.path.splitext(temp)[1] == '.tif':
    #         if temp.split('_')[0] < '20230611':
    #             print('continue')
    #             continue
    #         tif_name = os.path.join(tif_path, temp)
    #         date = temp.split('.')[0]
    #
    #         """
    #         blh_name = os.path.join(blh_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_blh_resample.tif')
    #         r_name = os.path.join(r_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_r_resample.tif')
    #         sp_name = os.path.join(sp_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_sp_resample.tif')
    #         tp_name = os.path.join(tp_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_tp_resample.tif')
    #         wd_name = os.path.join(wd_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_wd_resample.tif')
    #         ws_name = os.path.join(ws_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_ws_resample.tif')
    #         t2m_name = os.path.join(t2m_path, date[:4] + '-' + date[4:6] + '-' + date[6:8] + '_t2m_resample.tif')
    #         """
    #         blh_name = os.path.join(blh_path, date + '_blh_resample_re.tiff')
    #         r_name = os.path.join(r_path, date + '_r_resample_re.tiff')
    #         sp_name = os.path.join(sp_path, date + '_sp_resample_re.tiff')
    #         wd_name = os.path.join(wd_path, date + '_wd_resample_re.tiff')
    #         ws_name = os.path.join(ws_path, date + '_ws_resample_re.tiff')
    #         t2m_name = os.path.join(t2m_path, date + '_t2m_resample_re.tiff')
    #         outputpath2 = os.path.join(output, os.path.splitext(temp)[0] + '_EX_线性_month.tif')
    #         outputpath1 = r'J:\My_PM2.5Data\WuHai\Result'
    #         # outputpath = os.path.join(output, os.path.splitext(temp)[0] + '_machine_xgboost.tif')
    #
    #         machine_tif(models, second_model, tif_name, blh_name, r_name, sp_name, wd_name, ws_name, t2m_name, demPath, popPath, outputpath1, outputpath2)


    # outputPath = r'K:\231021 武汉成都\武汉PM10Result'
    # tifPaths = glob.glob(r'K:\231021 武汉成都\武汉_clip\*.tif')
    # for tifPath in tifPaths:
    #     # if os.path.basename(tifPath)[:-4] <= '20201109':
    #     #     continue
    #     date = os.path.basename(tifPath).split('_')[0]
    #     # strDate = str(date[0: 4]) + '-' + str(date[5: 7])
    #     strDate = str(date[0: 4]) + '-' + str(date[4: 6] + '-' + str(date[6: 8]))
    #     blh = os.path.join(r'K:\231021 武汉成都\武汉对应气象数据\重采样\blh', strDate + '_blh_resample.tif')
    #     r = os.path.join(r'K:\231021 武汉成都\武汉对应气象数据\重采样\r', strDate + '_r_resample.tif')
    #     sp = os.path.join(r'K:\231021 武汉成都\武汉对应气象数据\重采样\sp', strDate + '_sp_resample.tif')
    #     wd = os.path.join(r'K:\231021 武汉成都\武汉对应气象数据\重采样\wd', strDate + '_wd_resample.tif')
    #     ws = os.path.join(r'K:\231021 武汉成都\武汉对应气象数据\重采样\ws', strDate + '_ws_resample.tif')
    #     t2m = os.path.join(r'K:\231021 武汉成都\武汉对应气象数据\重采样\t2m', strDate + '_t2m_resample.tif')
    #     if date[0: 4] == '2020':
    #         pop = r"K:\231021 武汉成都\武汉市人口\resampling\2020_wuhan_clipresample2.tif"
    #     elif date[0: 4] == '2021':
    #         pop = r"K:\231021 武汉成都\武汉市人口\resampling\2021_wuhan_clipresample2.tif"
    #     elif date[0: 4] == '2022':
    #         pop = r"K:\231021 武汉成都\武汉市人口\resampling\2022_wuhan_clipresample2.tif"
    #     elif date[0: 4] == '2023':
    #         pop = r"K:\231021 武汉成都\武汉市人口\resampling\2022_wuhan_clipresample2.tif"
    #     dem = r"K:\231021 武汉成都\WHDEM\WuHan_dem_resample.tif"
    #     # try:
    #     machine_tif(models, second_model, tifPath, blh, r, sp, wd, ws, t2m, dem, pop, outputPath)

    ETPath = r"J:\AODinversion\2024大论文\ML\et\ET_AOD_0.m"

    et_model = joblib.load(ETPath)

    models = [et_model]

    tif_paths = glob.glob(r'\*.tif')

    blh_path = glob.glob(r'.\blh\*.tif')
    r_path = glob.glob(r'.\r\*.tif')
    sp_path = glob.glob(r'.\sp\*.tif')
    wd_path = glob.glob(r'.\wd\*.tif')
    ws_path = glob.glob(r'.\ws\*.tif')
    t2m_path = glob.glob(r'.\t2m\*.tif')

    output = r'J:\AODinversion\2024大论文\result\day\GF1'
    demPath = r"J:\机器学习测试文件\dem_tif\nber_dem_resample.tif"
    for tifPath in tif_paths:
        print(tifPath)
        date = os.path.basename(tifPath)
        blh_name = os.path.join(blh_path,
                               str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_blh_resample.tif')
        r_name = os.path.join(r_path,
                                str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_r_resample.tif')
        sp_name = os.path.join(sp_path,
                                str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_sp_resample.tif')
        wd_name = os.path.join(wd_path,
                              str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_wd_resample.tif')
        ws_name = os.path.join(ws_path,
                                str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_ws_resample.tif')
        t2m_name = os.path.join(t2m_path,
                              str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_t2m_resample.tif')


        machine_tif(models, None, tifPath, blh_name, r_name,
                    sp_name, wd_name, ws_name, t2m_name, demPath, None, output)


    """tif_paths = glob.glob(r'K:\2024大论文 AOD\TOA\landsat8_resample\*.tif')
    tco_path = r'K:\2024大论文 AOD\qx\2022tco'
    tcwv_path = r'K:\2024大论文 AOD\qx\2022tcwv'
    output = r'J:\AODinversion\2024大论文\result\day\Landsat8'
    demPath = r"J:\机器学习测试文件\dem_tif\nber_dem_resample.tif"
    for tifPath in tif_paths:
        print(tifPath)
        date = os.path.basename(tifPath)
        tconame = os.path.join(tco_path,
                               str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_tco3_resample.tif')
        tcwvname = os.path.join(tcwv_path,
                                str(date[0:4]) + '-' + str(date[4: 6]) + '-' + str(date[6: 8]) + '_tcwv_resample.tif')
        try:
            machine_AOD2(models, tifPath, tcwvname, tconame, demPath, output)
        except:
            continue"""