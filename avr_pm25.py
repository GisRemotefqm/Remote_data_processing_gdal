import pandas as pd
import numpy as np
import os
import datetime
import glob


def math_arvg(inputpath, outputpath):

    data = pd.read_csv(inputpath)
    data['Date (LT)'] = pd.to_datetime(data['Date (LT)'])
    result = data.resample('D', on='Date (LT)')['NowCast Conc.'].mean().reset_index()
    print(os.path.join(outputpath, os.path.basename(inputpath)[:-4] + '_mean.csv'))
    result.to_csv(os.path.join(outputpath, os.path.basename(inputpath)[:-4] + '_mean.csv'))


def math_daliy_PM25(csvPath, outputPath):

    df = pd.read_csv(csvPath)
    print(df.head())
    stations = df['station'].to_numpy()
    unqiue_stations = df['station'].drop_duplicates()
    unqiue_date = df['date'].drop_duplicates()
    print(unqiue_stations)
    stations = unqiue_stations.tolist()
    date = unqiue_date.tolist()
    result = []
    for i in range(len(stations)):
        for j in date:
            temp = df[(df['station'] == stations[i]) & (df['date'] == j)]
            if temp.size / 8 > 20:
        #aver = df.groupby(['date', 'stations']).mean()
                aver = temp.groupby(['station']).mean()
                result.append(aver)

    result = pd.concat(result)
    result.to_csv(outputPath, encoding='utf-8_sig')


if __name__ == '__main__':
    path = r"J:\23_06_lunwen\231017 只用尼泊尔境内数据\真实PM2.5数据\小时数据\231107 尼泊尔_PM2.5小时站点数据(添加了不更新的点)_unique_去除异常值.csv"

    math_daliy_PM25(path, r"J:\23_06_lunwen\231017 只用尼泊尔境内数据\真实PM2.5数据\231107 尼泊尔_PM2.5站点日平均数据(去除异常值)")

    # df = pd.read_csv(r"J:\23_06_lunwen\231017 只用尼泊尔境内数据\231103 ALL尼泊尔及其周边真实站点数据日平均_unqiue.csv")
    # # print(df.head())
    # # # df['stations']
    # df.drop_duplicates(subset=['station', 'lon', 'lat'], keep='first', inplace=True, ignore_index=False)
    # df.to_csv(r"J:\23_06_lunwen\231017 只用尼泊尔境内数据\231103 231103尼泊尔及其周边真实站点_unqiue.csv", index=False)