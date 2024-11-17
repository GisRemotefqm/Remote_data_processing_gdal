import time
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from subprocess import call


def get_everyday(year, begin_date, end_date):
    if isinstance(year, int) is False:
        raise FileNotFoundError("the type of year must be int!")
    everyday = []
    for i in range(1, 13):
        months = 100 + i
        if i == 2:
            if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
                days = 29
            else:
                days = 28
        elif i in {1, 3, 5, 7, 8, 10, 12}:
            days = 31
        else:
            days = 30
        for j in range(101, 101+days):
            y = str(year)
            month = str(months)[1:]
            day = str(j)[1:]
            everyday.append(y + month + day)

    assert begin_date in everyday and end_date in everyday, "begin_date or end_date is wrong!"
    begin_index = everyday.index(begin_date)
    end_index = everyday.index(end_date)
    return everyday[begin_index:end_index + 1]


def get_json_link():

    s = Service(r".\msedgedriver.exe")
    wd = webdriver.Edge(service=s)
    wd.implicitly_wait(10)
    wd.get('https://openaq-fetches.s3.amazonaws.com/index.html')

    realtime = wd.find_element(By.CSS_SELECTOR, '#tbody-s3objects > tr:nth-child(9) > td.sorting_2 > a')
    realtime.click()

    search = wd.find_element(By.CSS_SELECTOR, '#tb-s3objects_filter > label > input')
    search.send_keys('2021')

    # 需要点击最下方 next 按钮, 7次
    for page in range(1, 9):

        time.sleep(10)
        date_num = len(wd.find_elements(By.CSS_SELECTOR, '#tbody-s3objects > tr'))
        # 一个页面有 50 个日期
        for i in range(1, date_num+1):

            # 多次点击 next, 回到 日期对应的页面
            if page > 1:
                for c in range(1, page):
                    wd.find_element(By.CSS_SELECTOR, '#tb-s3objects_next > a').click()
                    time.sleep(3)

            time.sleep(15)
            link = []
            date_button = wd.find_element(
                By.CSS_SELECTOR,
                '#tbody-s3objects > tr:nth-child(' + str(i) + ') > td.sorting_2 > a')
            date = date_button.get_attribute('data-prefix').split('/')[1].split('-')
            date = date[0] + date[1] + date[2]
            out_path = 'E:/OpenAQ_PM25_2020/' + date + '_link.csv'

            date_button.click()
            time.sleep(5)

            # 页面的个数
            page_num = len(wd.find_elements(By.CSS_SELECTOR, '#tb-s3objects_paginate > ul > li')) - 2
            for n in range(1, page_num + 1):

                if n > 1:
                    next_bu = wd.find_element(By.CSS_SELECTOR, '#tb-s3objects_next > a')
                    next_bu.click()

                time.sleep(5)
                rows_num = len(wd.find_elements(By.CSS_SELECTOR, '#tbody-s3objects > tr'))
                print(rows_num)

                for j in range(1, rows_num + 1):
                    ndjson_f = wd.find_element(
                        By.CSS_SELECTOR,
                        '#tbody-s3objects > tr:nth-child(' + str(j) + ') > td.sorting_2 > a'
                    )
                    json_url = ndjson_f.get_attribute('href')
                    link.append(json_url)
                    nn = 1

            df = pd.DataFrame(link, columns=['link'])
            df.to_csv(path_or_buf=out_path, index=False)
            # 回到 /realtime/ 文件夹
            wd.find_element(By.CSS_SELECTOR, '#breadcrumb > li:nth-child(2) > a').click()
            time.sleep(10)


def download_by_idm(download_path, url_list):

    IDM = r"D:\Program Files\IDM\IDMan.exe"

    # /d URL  # 根据URL下载文件
    # /s  # 开始下载队列中的任务
    # /p  # 定义文件要存储在本地的地址
    # /f  # 定义文件存储在本地的文件名
    # /q  # 下载成功后IDM将退出。
    # /h  # 下载成功后IDM将挂起你的链接
    # /n  # 当IDM不出问题时启动静默模式
    # /a  # 添加指定文件到/d的下载队列，但是不进行下载

    for url in url_list:
        call([IDM, '/d', url, '/p', download_path, '/a'])

    call([IDM, '/s'])


if __name__ == '__main__':

    # get_json_link()

    day_list = get_everyday(2020, '20200729', '20201231')
    for day in day_list:
        print(day)
        link_df = pd.read_csv('./2020_download_link/' + day + '_link.csv')
        link_list = link_df['link'].to_list()
        download_by_idm('./ndjson/' + day, link_list)
        time.sleep(300)


