import os.path
import cdsapi

niboer = [31, 79, 25, 91]
asaibaijiang = [42.7, 43, 37.3, 51]
sililanka = [11, 79, 5.49, 82]
kenniya = [5.2, 33.2, -5.2, 42.3]
five_country = [29, 90, 4.1, 111]

nber_AOD = [32, 74, 22, 95]
kny_AOD = [5.2, 29, -5.2, 42.3]
Nanjing = [33, 117, 29, 120]
ChenDu = [32, 102, 29, 106]
WuHan = [33, 112, 28, 116]

riben = [41, 127, 29, 145]
qenbl = [46, 25, 56, 35]
sss = [35, -80, 45, -70]
country = [[30,13,34,17]]
countryName = ['SuDan']

# 下载t2m
"""for countryArg in range(len(country)):

    for i in range(2001, 2011):
        for j in range(3, 4):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-land',

                {
                    # 'product_type': 'reanalysis',

                    'variable': '2m_temperature',
                    # 'pressure_level': '975',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                    'format': 'netcdf.zip',
                },
                r'L:\福岛\\' + countryName[countryArg] + '\\t2m\\' + str(i) + str(j).rjust(2, '0') + 'download.netcdf.zip')"""

# 下载温度数据
c = cdsapi.Client()
hpa_list = ['400', '500', '600', '700', '850']


"""for hpa in hpa_list:
    for year in range(1970, 1980):
        for month in range(2, 5):
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': 'temperature',
                    'month': [str(month)],
                    'year': str(year),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [
                        41, 127, 29, 145,
                    ],
                    'pressure_level': hpa,
                },
                r'J:\福岛\ERA5三哩岛\大气温度\\' + '\\' + hpa + '\\' + 't' + str(year) + str(month) + hpa + 'download.nc')"""


"""c = cdsapi.Client()
for hpa in hpa_list:
    for year in range(1970, 1980):
        for month in range(2, 5):
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': 'relative_humidity',
                    'month': [str(month)],
                    'year': str(year),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [
                        41, 127, 29, 145,
                    ],
                    'pressure_level': hpa,
                },
                r'J:\福岛\ERA5三哩岛\相对湿度\\' + hpa + '\\' + 'rh' + str(year) + str(month) + hpa + 'download.nc')"""



# for hpa in hpa_list:
#     for year in range(1977, 1987):
#         for month in range(3, 6):
#             c.retrieve(
#                 'reanalysis-era5-pressure-levels',
#                 {
#                     'product_type': 'reanalysis',
#                     'format': 'netcdf',
#                     'variable': 'relative_humidity',
#                     'month': [str(month)],
#                     'year': str(year),
#                     'day': [
#                         '01', '02', '03',
#                         '04', '05', '06',
#                         '07', '08', '09',
#                         '10', '11', '12',
#                         '13', '14', '15',
#                         '16', '17', '18',
#                         '19', '20', '21',
#                         '22', '23', '24',
#                         '25', '26', '27',
#                         '28', '29', '30',
#                         '31',
#                     ],
#                     'time': [
#                         '00:00', '01:00', '02:00',
#                         '03:00', '04:00', '05:00',
#                         '06:00', '07:00', '08:00',
#                         '09:00', '10:00', '11:00',
#                         '12:00', '13:00', '14:00',
#                         '15:00', '16:00', '17:00',
#                         '18:00', '19:00', '20:00',
#                         '21:00', '22:00', '23:00',
#                     ],
#                     'area': [
#                         46, 25, 56, 35,
#                     ],
#                     'pressure_level': hpa,
#                 },
#                 r'J:\福岛\ERA5切尔诺贝利\相对湿度\\' + '\\' + hpa + '\\' + 'rh' + str(year) + str(
#                     month) + hpa + 'download.nc')
#
#
# c = cdsapi.Client()
# for hpa in hpa_list:
#     for year in range(1977, 1987):
#         for month in range(3, 6):
#             c.retrieve(
#                 'reanalysis-era5-pressure-levels',
#                 {
#                     'product_type': 'reanalysis',
#                     'format': 'netcdf',
#                     'variable': 'temperature',
#                     'month': [str(month)],
#                     'year': str(year),
#                     'day': [
#                         '01', '02', '03',
#                         '04', '05', '06',
#                         '07', '08', '09',
#                         '10', '11', '12',
#                         '13', '14', '15',
#                         '16', '17', '18',
#                         '19', '20', '21',
#                         '22', '23', '24',
#                         '25', '26', '27',
#                         '28', '29', '30',
#                         '31',
#                     ],
#                     'time': [
#                         '00:00', '01:00', '02:00',
#                         '03:00', '04:00', '05:00',
#                         '06:00', '07:00', '08:00',
#                         '09:00', '10:00', '11:00',
#                         '12:00', '13:00', '14:00',
#                         '15:00', '16:00', '17:00',
#                         '18:00', '19:00', '20:00',
#                         '21:00', '22:00', '23:00',
#                     ],
#                     'area': [
#                         46, 25, 56, 35,
#                     ],
#                     'pressure_level': hpa,
#                 },
#                 r'J:\福岛\ERA5切尔诺贝利\大气温度\\' + hpa + '\\' + 't' + str(year) + str(
#                     month) + hpa + 'download.nc')

# 下载t2m 陆地和海洋都有的
"""for countryArg in range(len(country)):

    for i in range(1976, 1987):
        for j in range(4, 6):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-single-levels',

                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': '2m_temperature',
                    # 'pressure_level': '975',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                    # 'format': 'netcdf.zip',
                },
                r'H:\福岛\\' + countryName[countryArg] + '\\t2m\\' + str(i) + str(j).rjust(2, '0') + 'download.nc')"""


# 下载sp 陆地和海洋都有的
"""for countryArg in range(len(country)):
    for i in range(2003, 2012):
        for j in range(3, 4):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': 'surface_pressure',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                    'format': 'netcdf',
                },
                r'L:\福岛\\' + countryName[countryArg] +'\\sp\\' + str(i) + str(j).rjust(2, '0') + 'download.netcdf.zip')
"""

"""
# 下载10_u
for countryArg in range(len(country)):
    for i in range(2018, 2023):
        for j in range(1, 13):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': '10m_u_component_of_wind',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg],

                    'format': 'netcdf.zip',
                },
                r'J:\23_06_lunwen\231102 气象数据\\' + countryName[countryArg] + '\\10_u\\' + str(i) + str(j).rjust(2, '0') + 'download.netcdf.zip')
"""


# 10_v
"""for countryArg in range(len(country)):
    for i in range(2018, 2023):
        for j in range(1, 13):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': '10m_v_component_of_wind',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': 
                        country[countryArg],

                    'format': 'netcdf.zip',
                },
                r'J:\23_06_lunwen\231102 气象数据\\' + countryName[countryArg] +'\\10_v\\' + str(i) + str(j).rjust(2, '0') + 'download.netcdf.zip')"""

# sp
"""for countryArg in range(len(country)):
    for i in range(2003, 2012):
        for j in range(3, 4):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': 'surface_pressure',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                    'format': 'netcdf.zip',
                },
                r'L:\福岛\\' + countryName[countryArg] +'\\sp\\' + str(i) + str(j).rjust(2, '0') + 'download.netcdf.zip')"""


"""for countryArg in range(len(country)):
    for i in range(2011, 2023):
        for j in range(1, 13):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': 'surface_pressure',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                    'format': 'netcdf.zip',
                },
                r'J:\23_06_lunwen\231102 气象数据\\' + countryName[countryArg] +'\\sp\\' + str(i) + str(j).rjust(2, '0') + 'download.netcdf.zip')
     """

# tp
"""for countryArg in range(len(country)):
    for i in range(2023, 2024):
        for j in range(1, 7):
            print(i)
            print(str(j).rjust(2, '0'))

            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': 'total_precipitation',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,

                    'format': 'netcdf.zip',
                },
                r'J:\毕业论文\气象数据\\' + countryName[countryArg] + '\\tp\\' + str(i) + str(j).rjust(2, '0') + 'download.zip'
            )"""

# blh
# for countryArg in range(len(country)):
#     for i in range(2018, 2023):
#         for j in range(1, 13):
#             print(i)
#             print(str(j).rjust(2, '0'))
#             c = cdsapi.Client()
#
#             c.retrieve(
#                 'reanalysis-era5-single-levels',
#                 {
#                     'product_type': 'reanalysis',
#                     'format': 'netcdf',
#                     'variable': 'boundary_layer_height',
#                     'year': str(i),
#                     'month': str(j).rjust(2, '0'),
#                     'day': [
#                         '01', '02', '03',
#                         '04', '05', '06',
#                         '07', '08', '09',
#                         '10', '11', '12',
#                         '13', '14', '15',
#                         '16', '17', '18',
#                         '19', '20', '21',
#                         '22', '23', '24',
#                         '25', '26', '27',
#                         '28', '29', '30',
#                         '31',
#                     ],
#                     'time': [
#                         '00:00', '01:00', '02:00',
#                         '03:00', '04:00', '05:00',
#                         '06:00', '07:00', '08:00',
#                         '09:00', '10:00', '11:00',
#                         '12:00', '13:00', '14:00',
#                         '15:00', '16:00', '17:00',
#                         '18:00', '19:00', '20:00',
#                         '21:00', '22:00', '23:00',
#                     ],
#                     'area':
#                         country[countryArg]
#                     ,
#                 },
#
#                  r'J:\23_06_lunwen\231102 气象数据\\' + countryName[countryArg] + '\\blh\\' + str(i) + str(j).rjust(2, '0') + 'download.nc')
#

# rh


"""for countryArg in range(len(country)):
    for i in range(2003, 2011):
        for j in range(3, 4):
            print(i)
            print(str(j).rjust(2, '0'))
            c = cdsapi.Client()

            c.retrieve(
                 'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': 'relative_humidity',
                    'pressure_level': '900',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                },

                 r'L:\福岛\\' + countryName[countryArg] + '\\rh\\' + str(i) + str(j).rjust(2, '0') + 'download.nc')


# rh
for countryArg in range(len(country)):
    for i in range(2011, 2012):
        for j in range(1, 13):
            print(i)
            print(str(j).rjust(2, '0'))
            c = cdsapi.Client()

            c.retrieve(
                 'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': 'relative_humidity',
                    'pressure_level': '900',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                },

                 r'L:\福岛\\' + countryName[countryArg] + '\\rh\\' + str(i) + str(j).rjust(2, '0') + 'download.nc')"""

# tto
for countryArg in range(len(country)):
    for i in range(2023, 2024):
        for j in range(4, 5):
            print(i)
            print(str(j).rjust(2, '0'))
            print('K:\\BMXM\\' + countryName[countryArg] + '\\tcwv\\' + str(i) + str(j).rjust(2, '0') + 'download.nc')
            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    # 'pressure_level': '950',
                    'variable': 'total_column_water_vapour',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                },

                r'K:\BMXM\XMDataset\\' + countryName[countryArg] + '\\tcwv\\' + str(i) + str(j).rjust(2, '0') + 'download.nc')


for countryArg in range(len(country)):
    for i in range(2023, 2024):
        for j in range(4, 5):
            print(i)
            print(str(j).rjust(2, '0'))
            c = cdsapi.Client()

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    # 'pressure_level': '950',
                    'variable': 'total_column_ozone',
                    'year': str(i),
                    'month': str(j).rjust(2, '0'),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area':
                        country[countryArg]
                    ,
                },
                r'K:\BMXM\XMDataset\\' + countryName[countryArg] + '\\tco\\' + str(i) + str(j).rjust(2, '0') + 'download.nc')




lanmei = [5, 90, 29, 110]


"""
    c = cdsapi.Client()

c.retrieve(
    'satellite-sea-surface-temperature',
    {
        'version': '2_1',
        'variable': 'all',
        'format': 'zip',
        'processinglevel': 'level_3c',
        'sensor_on_satellite': 'avhrr_on_noaa_17',
        'year': '2003',
        'month': '01',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
    },
    'download.zip')
"""