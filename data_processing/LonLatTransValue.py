import osgeo.osr as osr
import numpy as np

def get_lon_lat(data_list, transform, projection, location):
    """

    :param data_list: 像元值矩阵
    :param transform: 仿射变化系数
    :param projection: 投影
    :param location: 经纬度，这个得是np数组
    :return:
    """
    data_list = np.array(data_list)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)

    srs_lat_lon = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs_lat_lon, srs)

    new_location = [None, None]
    pixel_value_list = []
    for i in location.shape[1]:
        new_location[1], new_location[2], holder = ct.TransformPoint(location[i, 0], location[i, 1])

        x = (new_location[1] - transform[0]) / transform[1]
        y = (new_location[0] - transform[2]) / transform[5]

        pixel_value_list.append(data_list[:, x, y])

    return pixel_value_list
