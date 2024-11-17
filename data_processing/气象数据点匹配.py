import numpy as np
import pandas as pd
from xgboost import XGBRegressor

lon = [49.53, 79.8483, 36.81056, 85.335]
lat = [40.22, 6.913056, -1.234167, 27.73833]
country_name = ['asbk', 'sllk', 'kny', 'nber']

kdmd_25_20 = r'H:\地面无异常值\地面无异常值\PhoraDurbarKathmandu_PM2.5_2020_YTD_mean.csv'
kdmd_25_21 = r'H:\地面无异常值\地面无异常值\PhoraDurbarKathmandu_PM2.5_2021_YTD_mean.csv'
kdmd_25_22 = r'H:\地面无异常值\地面无异常值\PhoraDurbarKathmandu_PM2.5_2022_YTD_mean.csv'

bk_25_20 = r'H:\地面无异常值\地面无异常值\Baku_PM2.5_2020_YTD_mean.csv'
bk_25_21 = r'H:\地面无异常值\地面无异常值\Baku_PM2.5_2021_YTD_mean.csv'
bk_25_22 = r'H:\地面无异常值\地面无异常值\Baku_PM2.5_2022_YTD_mean.csv'

clb_25_20 = r'H:\地面无异常值\地面无异常值\Colombo_PM2.5_2020_YTD_mean.csv'
clb_25_21 = r'H:\地面无异常值\地面无异常值\Colombo_PM2.5_2021_YTD_mean.csv'
clb_25_22 = r'H:\地面无异常值\地面无异常值\Colombo_PM2.5_2022_YTD_mean.csv'

nrb_25_20 = r'H:\地面无异常值\地面无异常值\Nairobi_PM2.5_2020_YTD_mean.csv'
nrb_25_21 = r'H:\地面无异常值\地面无异常值\Nairobi_PM2.5_2021_YTD_mean.csv'
nrb_25_22 = r'H:\地面无异常值\地面无异常值\Nairobi_PM2.5_2022_YTD_mean.csv'

nber = r'G:\KongTianYuan\四国气象表\尼泊尔.csv'
sllk = r'G:\KongTianYuan\四国气象表\斯里兰卡.csv'
asbj = r'G:\KongTianYuan\四国气象表\阿塞拜疆.csv'
kny = r'G:\KongTianYuan\四国气象表\肯尼亚.csv'

n_data = pd.read_csv(nber, encoding='utf-8')
s_data = pd.read_csv(sllk, encoding='utf-8')
a_data = pd.read_csv(asbj, encoding='utf-8')
k_data = pd.read_csv(kny, encoding='utf-8')

all_date = k_data['日期']
try:
    nber_pm = pd.read_csv(kdmd_25_22, encoding='gbk')
except:
    nber_pm = None

try:
    asbj_pm = pd.read_csv(bk_25_22, encoding='gbk')
except:
    asbj_pm = None

try:
    sllk_pm = pd.read_csv(clb_25_22, encoding='gbk')
except:
    sllk_pm = None

try:
    kny_pm = pd.read_csv(nrb_25_22, encoding='gbk')
except:
    kny_pm = None

data = pd.read_csv(r'H:\step3\GF1_四国WFV_6匹配_2022.csv', encoding='utf-8')

data_lon = data.lon.tolist()
data_lat = data.lat.tolist()
result_list = []
for i in range(len(data_lon)):

    if data_lon[i] == lon[0]:

        temp_date = data['date'][i]
        arg = np.argwhere(a_data['日期'].to_numpy() == temp_date)
        if asbj_pm is None:
            result_list.append(np.hstack((a_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                      'band5', 'band6', 'band7',
                                                                                      'band8',
                                                                                      'lon', 'lat']].iloc[i, :], 0)))
        else:
            pm_date = np.argwhere(asbj_pm['Date (LT)'].to_numpy() == temp_date)

            if pm_date.size == 0:

                result_list.append(np.hstack((a_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                          'band5', 'band6', 'band7',
                                                                                          'band8',
                                                                                          'lon', 'lat']].iloc[i, :], 0)))
            else:
                result_list.append(np.hstack((a_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                          'band5', 'band6', 'band7',
                                                                                          'band8',
                                                                                          'lon', 'lat']].iloc[i, :],
                                              asbj_pm['NowCast Conc.'].iloc[pm_date[0, 0]])))

    if data_lon[i] == lon[1]:

        temp_date = data['date'][i]
        arg = np.argwhere(s_data['日期'].to_numpy() == temp_date)
        if sllk_pm is None:
            result_list.append(np.hstack((s_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                      'band5', 'band6', 'band7',
                                                                                      'band8',
                                                                                      'lon', 'lat']].iloc[i, :], 0)))
        else:

            pm_date = np.argwhere(sllk_pm['Date (LT)'].to_numpy() == temp_date)
            if pm_date.size == 0:

                result_list.append(np.hstack((s_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                          'band5', 'band6', 'band7',
                                                                                          'band8',
                                                                                          'lon', 'lat']].iloc[i, :], 0)))
            else:
                result_list.append(np.hstack((s_data.iloc[arg[0, 0], :].to_numpy(), data[
                                                                                        ['band1', 'band2', 'band3', 'band4',
                                                                                         'band5', 'band6', 'band7',
                                                                                         'band8',
                                                                                         'lon', 'lat']].iloc[i, :],
                                              sllk_pm['NowCast Conc.'].iloc[pm_date[0, 0]])))

    if data_lon[i] == lon[2]:

        temp_date = data['date'][i]
        arg = np.argwhere(k_data['日期'].to_numpy() == temp_date)
        if kny_pm is None:
            result_list.append(np.hstack((k_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                      'band5', 'band6', 'band7',
                                                                                      'band8',
                                                                                  'lon', 'lat']].iloc[i, :], 0)))
        else:
            pm_date = np.argwhere(kny_pm['Date (LT)'].to_numpy() == temp_date)
            if pm_date.size == 0:
                result_list.append(np.hstack((k_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                          'band5', 'band6', 'band7',
                                                                                          'band8',
                                                                                      'lon', 'lat']].iloc[i, :], 0)))
            else:
                result_list.append(np.hstack((k_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                          'band5', 'band6', 'band7',
                                                                                          'band8',
                                                                           'lon', 'lat']].iloc[i, :],
                               kny_pm['NowCast Conc.'].iloc[pm_date[0, 0]])))

    if data_lon[i] == lon[3]:

        temp_date = data['date'][i]
        arg = np.argwhere(n_data['日期'].to_numpy() == temp_date)
        if nber_pm is None:
            result_list.append(np.hstack((n_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                      'band5', 'band6', 'band7',
                                                                                      'band8',
                                                                                  'lon', 'lat']].iloc[i, :], 0)))
        else:
            pm_date = np.argwhere(nber_pm['Date (LT)'].to_numpy() == temp_date)
            if pm_date.size == 0:

                result_list.append(np.hstack((n_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                          'band5', 'band6', 'band7',
                                                                                          'band8',
                                                                                      'lon', 'lat']].iloc[i, :], 0)))
            else:
                result_list.append(np.hstack((n_data.iloc[arg[0, 0], :].to_numpy(), data[['band1', 'band2', 'band3', 'band4',
                                                                                          'band5', 'band6', 'band7',
                                                                                          'band8',
                                                                           'lon', 'lat']].iloc[i, :],
                               nber_pm['NowCast Conc.'].iloc[pm_date[0, 0]])))

colunms = ['日期', 'blh', 'r', 'sp', 't2m', 'tp', 'wd', 'ws', 'band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7', 'band8', 'lon', 'lat', 'pm2.5']
shuju = pd.DataFrame(np.array(result_list), columns=colunms)
shuju.to_csv(r'H:\step3\GF-1_22_匹配结果.csv', index=False, encoding='gbk')