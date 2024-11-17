# --coding:utf-8--
import pandas as pd
import numpy as np
import osgeo.gdal as gdal

from DataMath import ReadWrite_h5
import os

def gf_to_txt(gfpath, lon, lat, outputpath):


    gf_inputdir = os.listdir(gfpath)
    result_list = []

    for date_name in gf_inputdir:
        temp_path = os.path.join(gfpath, date_name)
        for data_name in os.listdir(temp_path):

            namestart, nameend = os.path.splitext(data_name)
            if nameend == '.tiff':
                tif_path = os.path.join(temp_path, data_name)
                print(tif_path)
                dataset = ReadWrite_h5.get_tifDataset(tif_path)
                imgx, imgy = ReadWrite_h5.get_RasterXY(dataset)
                date = namestart.split('_')[0]

                if dataset is None:
                    print('数据集出现错误')
                    continue
                projection, transform = ReadWrite_h5.get_GeoInformation(dataset)
                for i in range(len(lon)):
                    gf_x, gf_y = ReadWrite_h5.geo2imagexy(lon[i], lat[i], transform)
                    if ((gf_x <= imgx) & (gf_x >= 0)) & ((gf_y <= imgy) & (gf_y >= 0)):
                        # gf影像中的表观反射率的值
                        fanshe = ReadWrite_h5.get_RasterArr(dataset=dataset, lt_x=int(gf_x), lt_y=int(gf_y), rt_x=1, rt_y=1)
                        print(int(gf_x), int(gf_y))

                        result = np.hstack((fanshe.reshape(-1), lon[i], lat[i], np.array(os.path.basename(tif_path)[:-4])))
                        result_list.append(result)
                    else:
                        print(int(gf_x), int(gf_y))
                        continue
            else:
                continue
    pd_colunms = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7', 'band8', 'lon', 'lat', 'tif_name']
    pd_colunms = ['PM2.5', 'lon', 'lat', 'tif_name']
    shujuji = np.array(result_list)
    print(shujuji.shape)

    shujuji = pd.DataFrame(shujuji, columns=pd_colunms)
    shujuji.to_csv(outputpath, encoding='utf-8', index=False)

if __name__ == '__main__':

    aqi_date = pd.read_csv(r"C:\Users\fuqiming\Desktop\文件\比赛说明文件\AOD_550nm\jjj_station.csv", encoding='utf8')
    lon_lat = aqi_date[['lon', 'lat']].to_numpy()
    unqiue_lonlat = np.array(list(set([tuple(t) for t in lon_lat])))
    print(unqiue_lonlat)
    lon = unqiue_lonlat[:, 1].tolist()
    lat = unqiue_lonlat[:, 0].tolist()


    gf_to_txt(r'G:\大气-7.5\大气\从容应队\1\初赛\1_从容应队\1-从容应队\result', lon, lat, r'G:\大气-7.5\大气\从容应队\1\初赛\1_从容应队\1-从容应队\result\targetPredict.csv')