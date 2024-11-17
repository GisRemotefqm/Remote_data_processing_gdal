import math
import os
import numpy as np
from datetime import datetime, timedelta
import time

import pandas as pd

PI = math.pi

"""
python中时间日期格式化符号：

%y 两位数的年份表示（00-99）

%Y 四位数的年份表示（000-9999）

%m 月份（01-12）

%d 月内中的一天（0-31）

%H 24小时制小时数（0-23）

%I 12小时制小时数（01-12） 

%M 分钟数（00=59）

%S 秒（00-59）
--------------------------------
%a 本地简化星期名称

%A 本地完整星期名称

%b 本地简化的月份名称

%B 本地完整的月份名称

%c 本地相应的日期表示和时间表示

%j 年内的一天（001-366）

%p 本地A.M.或P.M.的等价符

%U 一年中的星期数（00-53）星期天为星期的开始

%w 星期（0-6），星期天为星期的开始

%W 一年中的星期数（00-53）星期一为星期的开始

%x 本地相应的日期表示

%X 本地相应的时间表示

%Z 当前时区的名称

%% %号本身 

"""


def Math_Angle(lat, dec, t):
    """
    :param lat: 纬度
    :param dec: 太阳赤纬
    :param t: 太阳时
    :return:
    """

    Hs = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(t)
    As = (np.sin(Hs) * np.sin(lat) - np.sin(dec)) / (np.cos(Hs) * np.cos(lat))

    return Hs, As


def math_dec(dates, diff_path):
    """

    :param day: 一年中的第几天
    :return: 太阳赤纬角,太阳时
    """

    dec = []
    t = []
    range_date = pd.period_range(start='2000-01-01', end='2000-12-31', freq='D')
    t_difference = pd.read_csv(diff_path, index_col=range_date)

    for date in dates:

        now_time = time.strptime(date, '%Y-%m-%d %H:%M:%S')
        yday = now_time.tm_yday
        b = PI * 2 * (yday - 1) / 365
        dec.append(0.006918 - 0.399912 * np.cos(b) + 0.070257 * np.sin(b)\
              - 0.006758* np.cos(2 * b) + 0.000907 * np.sin(2 * b) -\
              0.002697 * np.cos(3 * b) + 0.00148 * np.sin(3 * b))
        month, day = now_time.tm_mon, now_time.tm_mday
        str_date = '2000-' + str(month) + '-' + str(day)
        d_time, chara = t_difference.loc[str_date]['time'], t_difference.loc[str_date]['character']
        d_time = time.strptime(d_time, '%H:%M:%S')
        date_hour, date_min, date_second = now_time.tm_hour, now_time.tm_min, now_time.tm_sec

        d_second = date_hour * 60 * 60 + date_min * 60 + date_second
        now_second = now_time.tm_hour * 60 * 60 + now_time.tm_min * 60 + now_time.tm_sec
        if chara == '-':
            t_angle  = (((now_second - d_second) / 36000) - 12) * 15
            t.append(t_angle)

        elif chara == '+':
            t_angle = (((now_second + d_second) / 36000) - 12) * 15
            t.append(t_angle)

    return np.array(dec), np.array(t)


if __name__ == '__main__':
    range_date = pd.period_range(start='2000-01-01', end='2000-12-31', freq='D')