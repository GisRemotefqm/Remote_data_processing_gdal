import os
import pandas as pd
import numpy as np
from DataMath import ReadWrite_h5

country = ['尼泊尔', '阿塞拜疆', '肯尼亚', '斯里兰卡']
qx_list = ['blh', 'r', 'sp', 'tp', 't2m', 'wd', 'ws']
path = r'G:\KongTianYuan\四国tif数据'
lon = [49.53, 79.8483, 36.81056, 85.335]
lat = [40.22, 6.913056, -1.234167, 27.73833]
data_list = []
for country_name in country:

    country_path = os.path.join(path, country_name)
    country_list = []
    for qx_name in os.listdir(country_path):

        qx_path = os.path.join(country_path, qx_name)
        qx_list = []
        for data_name in os.listdir(qx_path):
            if os.path.splitext(data_name)[1] != '.tif':
                continue
            tif_path = os.path.join(qx_path, data_name)
            dataset = ReadWrite_h5.get_tifDataset(tif_path)
            if dataset is None:
                print('数据集出现错误')
                continue
            imgx, imgy = ReadWrite_h5.get_RasterXY(dataset)
            projection, transform = ReadWrite_h5.get_GeoInformation(dataset)
            # print(dataset)

            for i in range(len(lon)):
                gf_x, gf_y = ReadWrite_h5.geo2imagexy(lon[i], lat[i], transform)
                if ((gf_x <= imgx) & (gf_x >= 0)) & ((gf_y <= imgy) & (gf_y >= 0)):
                    # gf影像中的表观反射率的值
                    fanshe = ReadWrite_h5.get_RasterArr(dataset=dataset, lt_x=int(gf_x), lt_y=int(gf_y), rt_x=1, rt_y=1)
                    # print(int(gf_x), int(gf_y))

                    result = np.hstack((fanshe.reshape(-1), lon[i], lat[i], data_name[:10]))
                    qx_list.append(result)
                    data_list.append(result)
                else:
                    continue



print(data_list)

pd.DataFrame(np.array(data_list)).to_csv('单纯气象数据.csv')



