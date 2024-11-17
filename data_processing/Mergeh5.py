import h5py
import numpy as np

import ReadWrite_h5
import os

SCALE_FACTOR = 'Scale_Factor'
ADD_OFFSET = 'Add_Offset'
FILLVALUE = '_FillValue'


def get_GroupAndName(h5_filelist):

    dict = {}
    for datapath in h5_filelist:

        path = os.path.split(datapath[0])[0]
        datasetname = os.path.split(datapath[0])[-1]
        group = os.path.split(path)[-1]
        dict[datasetname] = group

    print(dict)
    return dict


def merge_h5(outputpath, group_dict, dataset_dict):

    with h5py.File(outputpath, 'a') as f:

        for datasetname, groupname in group_dict.items():
            try:
                group = f.create_group(groupname)
            except:
                pass
            for key, value in dataset_dict.items():

                if datasetname == key:
                    group.create_dataset(name=key, data=dataset_dict[key])


def get_fillvalue(h5_filelist, datasetname):

    for h5_file in h5_filelist:

        file_path, group_dname = h5_file[0].split('//')
        file_path = file_path.split('"')[1]
        group_name, dname = os.path.split(group_dname)
        if dname == datasetname:
            print('search fillvalue')
            h5_file = h5py.File(file_path)
            h5_group = h5_file[group_name]
            h5_dataset = h5_group[datasetname]
            fill_value = h5_dataset.attrs[FILLVALUE]

    return float(fill_value)


def get_h5Attribute(inputpath, group_name, datasetname):

    h5_file = h5py.File(inputpath)
    h5_group = h5_file[group_name]
    h5_dataset = h5_group[datasetname]
    scale_factor = h5_dataset.attrs[SCALE_FACTOR]
    add_offset = h5_dataset.attrs[ADD_OFFSET]

    return scale_factor, add_offset


def math_ObserveGeometry(add_offset, scale_factor, h5_filelist, datasetname):
    dataset = ReadWrite_h5.get_h5Dataset(h5_filelist, datasetname)
    observe_geometry = dataset.ReadAsArray()
    fill_value = get_fillvalue(h5_filelist, datasetname)
    observe_geometry[np.where(observe_geometry == fill_value)] = 0
    new_obgeo = scale_factor * (observe_geometry - add_offset)
    group_dict = {datasetname: new_obgeo}

    return group_dict


def get_datasetdict(h5_datalist, group_dict, noread_list):

    dataset_dict = {}
    for key in group_dict.keys():
        dataset = ReadWrite_h5.get_h5Dataset(h5_datalist, key)
        if key in noread_list:
            dataset_arr = 0
            dataset_dict[key] = dataset_arr
        else:
            dataset_arr = dataset.ReadAsArray()
            dataset_dict[key] = dataset_arr

    return dataset_dict


if __name__ == '__main__':

    inputpath = r'.\GF5_DPC_20200216_009441_L10000030552_B865.h5'
    Ireadname = 'I865P'
    sol_a_ang = 'Sol_Azim_Ang'
    sol_z_ang = 'Sol_Zen_Ang'
    v_azim_ang = 'View_Azim_Ang'
    v_zen_ang = 'View_Zen_Ang'
    recover_read_list = [sol_a_ang, sol_z_ang, v_zen_ang, v_azim_ang]

    # img_data = h5py.File(inputpath, 'r')

    h5_filelist = ReadWrite_h5.get_h5filelist(inputpath)

    group_dict = get_GroupAndName(h5_filelist)
    Idataset = ReadWrite_h5.get_h5Dataset(h5_filelist, Ireadname)
    dataset_dict = get_datasetdict(h5_filelist, group_dict, recover_read_list)

    for ob in recover_read_list:
        scale_factor, add_offset = get_h5Attribute(inputpath, 'Data_Fields', ob)
        ob_geo = math_ObserveGeometry(add_offset, scale_factor, h5_filelist, ob)
        dataset_dict.update(ob_geo)
    print(dataset_dict)
    merge_h5('outh5.h5', group_dict, dataset_dict)





