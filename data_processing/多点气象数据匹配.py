import numpy as np
import pandas as pd
import os


aqi_date = pd.read_csv(r'C:\Users\fuqiming\Desktop\Landsat卫星表观反射率表\多项式计算pm2.5.csv', encoding='gbk')
lon_lat = aqi_date[['lon', 'lat']].to_numpy()
unqiue_lonlat = np.array(list(set([tuple(t) for t in lon_lat])))
# print(unqiue_lonlat)
lon = unqiue_lonlat[:, 0].tolist()
lat = unqiue_lonlat[:, 1].tolist()
# print(lon, lat)


country_qx_csv = r'L:\小论文\231103 Nepal气象tif'
csv_name = ['南京GF1_多点.csv']

gf_1 = r'J:\毕业论文\PM2.5站点数据\2020-2022_Daliy_Mean_unqiue.csv'
gf_6 = r'C:\Users\fuqiming\Desktop\Landsat卫星表观反射率表\landsat9_汇总匹配.csv'

gf_data = pd.read_csv(gf_6)
gf_date = gf_data['date'].to_numpy()
gf_lon = gf_data['lon'].to_numpy()
gf_lat = gf_data['lat'].to_numpy()
print(gf_lat.shape, gf_lon.shape)
result_list = []
for csv in csv_name:
    csv_path = os.path.join(country_qx_csv, csv)
    fanshelv = pd.read_csv(csv_path)
    print(csv_path)

    for i in range(gf_date.shape[0]):
        temp_value = fanshelv[fanshelv['日期'] == gf_date[i]]

        value = temp_value[(temp_value['lon'] == gf_lon[i])].to_numpy()

        pm = aqi_date[(aqi_date['lon'] == gf_lon[i]) & (aqi_date['lon'] == gf_lon[i]) & (aqi_date['date'] == gf_date[i])]
        pm = pm['真实pm2.5'].to_numpy()

        if (gf_data.iloc[i, :].to_numpy().tolist() == []) | (value.tolist() == []) | (pm.tolist() == []):
            continue
        print(gf_data.iloc[i, :].to_numpy().shape, value.reshape(-1).shape, pm.reshape(-1).shape)
        # gf_data是反射率， value是气象数据， pm2.5数据
        # result = np.hstack((gf_data.iloc[i, :].to_numpy(), value.reshape(-1), pm.reshape(-1)))
        result = np.hstack((gf_data.iloc[i, :].to_numpy(), value.reshape(-1)))

        result_list.append(result)
    print('一次结束')


# acolunms = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7', 'band8', 'lon', 'lat', 'date', 'date_1', 'lon', 'lat', 'blh', 'r', 'sp', 't2m', 'tp', 'wd', 'ws', 'pm2.5']
acolunms = ['date', 'B2', 'B3', 'B4', 'B5', 'lon', 'lat', 'pm2.5', 'date_1', 'lon_1', 'lat_1', 'blh', 'r', 'sp', 't2m', 'tp', 'wd', 'ws']
print(np.array(result_list).shape)
pd.DataFrame(np.array(result_list), columns=acolunms).to_csv("landsat9_气象_反射率_PM2.5.csv", index=False)




