import cdsapi
niboer1 = [31, 79.5, 26.2, 88.3]
asaibaijiang1 = [42.7, 43, 37.3, 51]
sililanka1 = [9.6, 79, 5.49, 82]
kenniya1 = [5.2, 33.2, -5.2, 42.3]

city_fanwei1 = [[31, 79.5, 26.2, 88.3], [42.7, 43, 37.3, 51], [9.6, 79, 5.49, 82], [5.2, 33.2, -5.2, 42.3]]
cityname1 = ['nber', 'asbj', 'sllk', 'kny']

data_name = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'surface_pressure', 'total_precipitation']
xiangduishidu = 'relative_humidity'


for k in range(len(city_fanwei1)):
    print(len(city_fanwei1))
    for i in range(2020, 2023):
        for j in range(1, 13):

            print(i)
            print(str(j).rjust(2, '0'))
            print(cityname1[k])
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    # 相对适度的需要的
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    # -----
                    'variable': xiangduishidu,
                    'pressure_level': '950',
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
                            ' 21:00', '22:00', '23:00', ],
                    'area':
                            city_fanwei1[k]
                        ,

                        # 这块是多的数据集需要的
                        # 'format': 'netcdf.zip',


                },
                str(i) + str(j).rjust(2, '0') + cityname1[k] + '_' + 'rh_' + 'download.nc')


# 边界层高度

for k in range(len(city_fanwei1)):
    print(len(city_fanwei1))
    for i in range(2020, 2023):
        for j in range(1, 13):

            print(i)
            print(str(j).rjust(2, '0'))
            print(cityname1[k])
            c1 = cdsapi.Client()
            c1.retrieve(
                'reanalysis-era5-single-levels',
                {
                    # 相对湿度的需要的
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    # -----
                    'variable': 'boundary_layer_height',
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
                            ' 21:00', '22:00', '23:00', ],
                    'area':
                            city_fanwei1[k]
                        ,

                        # 这块是多的数据集需要的
                        # 'format': 'netcdf.zip',


                },
                str(i) + str(j).rjust(2, '0') + cityname1[k] + '_' + 'blh' + 'download.nc')



# 温度的数据
knyandnber = ['nber', 'kny']
kny_nber_fanwei = [[31, 79.5, 26.2, 88.3], [5.2, 33.2, -5.2, 42.3]]

for k in range(len(kny_nber_fanwei)):
    print(len(kny_nber_fanwei))
    for i in range(2020, 2023):
        for j in range(1, 13):
            print(i)
            print(str(j).rjust(2, '0'))
            print(knyandnber[k])
            c2 = cdsapi.Client()
            c2.retrieve(
                'reanalysis-era5-land',
                {

                    'variable': '2m_temperature',
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
                            ' 21:00', '22:00', '23:00', ],
                    'area':
                        kny_nber_fanwei[k]
                        ,


                    'format': 'netcdf.zip',


                },
                str(i) + str(j).rjust(2, '0') + knyandnber[k] + 'download.netcdf.zip')