import glob

import numpy as np
import torch
import torch.nn as nn
import random
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm
import pandas as pd
from DBN import DBN
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from osgeo import gdal
from GetData import GetImageData
standardScaler = StandardScaler()
Radius = 6371.009  # 地球半径, 单位km
h = 50  # window size, 单位为km
device = torch.device('cpu')
dbn_layers = [128, 64, 1]
# layers = [10, 10, 1]
from DataMath import ReadWrite_h5

def get_file_list(file_path, file_type):
    file_path = file_path
    file_list = []
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(file_type + " files not find in {}".format(file_path))
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == '.' + file_type:
                file_list.append([file, os.path.join(root, file)])
    return file_list


# def load_data(csv_path, ps_name, ps_dist_num, month=False):
#
#     if month:
#         pm25_name = 'month_mean'
#     else:
#         pm25_name = 'daily_mean'
#
#     csv_data = pd.read_csv(filepath_or_buffer=csv_path)
#     valid_data = csv_data.dropna(axis=0)  # 剔除nan
#
#     date_list = valid_data['date'].to_numpy(dtype='int64')
#     lat_list = valid_data['lat'].to_numpy(dtype='float32')
#     lon_list = valid_data['lon'].to_numpy(dtype='float32')
#
#     target_data = valid_data[pm25_name].to_numpy(dtype='float32')
#
#     # 时间编码
#     nd_pt = np.array(valid_data['t_code_x'])
#     nd_year = np.array(valid_data['year_code'])
#
#     aod_nd = np.array(valid_data['aod_13'])
#     blh_nd = np.array(valid_data['blh_13'])
#     rh_nd = np.array(valid_data['rh_13'])
#     t_nd = np.array(valid_data['t_13'])
#     ws_nd = np.array(valid_data['ws_13'])
#     wd_nd = np.array(valid_data['wd_13'])
#     sp_nd = np.array(valid_data['sp_13'])
#     ndvi_nd = np.array(valid_data['ndvi_13'])
#     lct_nd = np.array(valid_data['lct_13'])
#     lct_nd = np.round(lct_nd)
#     dem_nd = np.array(valid_data['dem_13'])
#     pop_nd = np.array(valid_data['pop_13'])
#
#     ps_nd = np.array(valid_data[ps_name])
#     ps_dist_nd = np.array(valid_data[ps_dist_num])
#     # ps_num_nd = np.array(valid_data['ps_num'])
#
#     contents_data = np.stack((
#         aod_nd, blh_nd, rh_nd, t_nd, ws_nd, wd_nd, sp_nd, ndvi_nd, lct_nd, pop_nd, dem_nd, ps_nd, ps_dist_nd), axis=-1)
#
#     max_v = np.amax(contents_data, axis=0)  # 每个特征的最大值, 数据归一化的分母
#     contents_data = contents_data/max_v  # 数据归一化
#
#     contents_data = np.insert(contents_data, contents_data.shape[1], nd_pt, axis=1)
#     contents_data = np.insert(contents_data, contents_data.shape[1], nd_year, axis=1)
#     contents_data = contents_data.astype('float32')
#
#     return date_list, lat_list, lon_list, contents_data, target_data
def LoadCSV(csvPath):

    df = pd.read_csv(csvPath, encoding='utf-8_sig');
    df2 = df[['band3', 'band4']]
    df = df[['blh', 'r', 'sp', 't2m', 'wd', 'ws', 'band1', 'band2', 'pd', 'pd_dist', 'DEM', 'PM2.5']];

    # , 'tcwv', 'tco', 'EVI'
    X = df.iloc[:, :-1];
    ndvi = (df2['band4'].to_numpy() - df2['band3'].to_numpy()) / (df2['band4'].to_numpy() + df2['band3'].to_numpy())
    ndvi = np.nan_to_num(ndvi, nan=0)

    X['ndvi'] = ndvi
    X = df.iloc[:, :-1].to_numpy();
    Y = df.iloc[:, -1].to_numpy();

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)

    y_train1 = y_train[:, np.newaxis]
    print(np.hstack((x_train, y_train1)).shape)
    pd.DataFrame(np.hstack((x_train, y_train1)), columns=['blh', 'r', 'sp', 't2m', 'wd', 'ws', 'band1', 'band2', 'pd', 'pd_dist', 'DEM', 'PM2.5']).to_csv(
        r'J:\毕业论文\Deep Learning\csv\daily_GFALL_train.csv', index=False)
    y_test1 = y_test[:, np.newaxis]
    pd.DataFrame(np.hstack((x_test, y_test1)), columns=['blh', 'r', 'sp', 't2m', 'wd', 'ws', 'band1', 'band2', 'pd', 'pd_dist', 'DEM', 'PM2.5']).to_csv(
        r'J:\毕业论文\Deep Learning\csv\daily_GFALL_PM2.5_test.csv', index=False)

    standardScaler.fit(x_train)

    x_train = standardScaler.transform(x_train)
    x_test = standardScaler.transform(x_test)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    x_train = x_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    x_test = x_test.to(torch.float32)
    y_test = y_test.to(torch.float32)

    return x_train, y_train, x_test, y_test, standardScaler;


def getStandardScaler(trainPath):
    trainDf = pd.read_csv(trainPath, encoding='gbk');
    trainDf = trainDf[['B2', 'B3', 'B4', 'B5',
             'Site_Elevation(m)', 'tcwv', 'tco', 'AOD_550nm']];

    x_train = trainDf.iloc[:, :-1].to_numpy();

    trainStandardScaler = StandardScaler()
    trainStandardScaler.fit(x_train)

    return trainStandardScaler

class MyDataset(Dataset):
    def __init__(self, contents_data, target_data):

        self.contents_data, self.target_data = contents_data, target_data

    def __len__(self):
        return self.target_data.shape[0]

    def __getitem__(self, index):


        contents, target = self.contents_data[index], self.target_data[index]
        contents_tensor = contents.clone()

        return contents_tensor, target


def dbn_pretrain(train_dataloader, dbn_layers, save_path, load_save=False):

    contents_data = train_dataloader.dataset.contents_data
    dbn = DBN(contents_data.shape[1], dbn_layers, savefile=save_path)

    if not load_save:
        dbn.train_DBN(torch.tensor(contents_data))  # 预训练权重W 与 偏置hb, vb
        model = dbn.initialize_model()  # 将得到的权重与偏置 赋予 全连接网络
    else:
        model = dbn.load_pretrained_model()  # dbn已经过训练, 直接将 savefile 加载到model
    return model


def train_one_epoch(dbn_model, optim, data_loader):
    criterion = nn.MSELoss()
    criterion.to(device)

    running_loss = 0
    data_size = len(data_loader.dataset)
    all_tar = torch.Tensor([[0]]).to(device)
    all_pred = torch.Tensor([[0]]).to(device)

    for inputs, target in data_loader:
        inputs, target = inputs.to(device), target.to(device)
        target = target.unsqueeze(1)
        optim.zero_grad()

        output = dbn_model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        running_loss += loss.item()

        all_tar = torch.cat((all_tar, target), 0)
        all_pred = torch.cat((all_pred, output), 0)

    running_loss /= data_size

    all_x = all_tar.cpu().detach().numpy().flatten()
    all_y = all_pred.cpu().detach().numpy().flatten()
    r2 = r2_score(all_x[1:], all_y[1:])

    return r2, running_loss


def evaluate(dbn_model, data_loader):
    criterion = nn.MSELoss()
    criterion.to(device)

    running_loss = 0
    data_size = len(data_loader.dataset)
    all_tar = torch.Tensor([[0]]).to(device)
    all_pred = torch.Tensor([[0]]).to(device)

    with torch.no_grad():
        for inputs, target in data_loader:
            inputs, target = inputs.to(device), target.to(device)
            target = target.unsqueeze(1)
            output = dbn_model(inputs)
            loss = criterion(output, target)
            running_loss += loss.item()

            all_tar = torch.cat((all_tar, target), 0)
            all_pred = torch.cat((all_pred, output), 0)

        running_loss /= data_size

        all_x = all_tar.cpu().detach().numpy().flatten()
        all_y = all_pred.cpu().detach().numpy().flatten()
        r2 = r2_score(all_x[1:], all_y[1:])

        return r2, running_loss


def train(cv_i, cv_type, dbn_model, optim, epochs, resume, check_path, save_path, train_dataloader, val_dataloader):
    start_epoch = -1
    if resume:
        path_checkpoint = check_path  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        dbn_model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    training = trange(start_epoch + 1, start_epoch + 1 + epochs)
    train_r2, train_loss = [], []
    val_r2, val_loss = [], []

    for epoch in training:

        t_r2, t_loss = train_one_epoch(dbn_model, optim, train_dataloader)
        train_r2.append(t_r2)
        train_loss.append(t_loss)

        v_r2, v_loss = evaluate(dbn_model, val_dataloader)
        val_r2.append(v_r2)
        val_loss.append(v_loss)

        training.set_description(
            str({'epoch': epoch + 1,
                 'train_r2': round(t_r2, 3), 'train_loss': round(t_loss, 4),
                 'val_r2': round(v_r2, 3), 'val_loss': round(v_loss, 4)}))
        if (epoch + 1) >= 150:
            checkpoint = {
                'epoch': epoch,
                'model': dbn_model.state_dict(),
                'optimizer': optim.state_dict(),
            }
            # i是用来标记交叉验证的
            torch.save(
                checkpoint,
                save_path + '/checkpoints/' + cv_type + '/cv_' + str(cv_i) + '/checkpoint_' + str(epoch + 1) + '.pth')

    d = {'train_r2': train_r2, 'train_loss': train_loss, 'val_r2': val_r2, 'val_loss': val_loss}
    df_r2_loss = pd.DataFrame(d)
    csv_path = save_path + '/r2_loss/' + cv_type + '/' + str(cv_i) + '_' + \
               str(start_epoch + 2) + '_' + str(start_epoch + 1 + epochs) + '_r2_loss.csv'
    df_r2_loss.to_csv(path_or_buf=csv_path, index=False)


def test(dbn_model, check_path, test_dataloader):
    path_checkpoint = check_path  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    dbn_model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数

    criterion = nn.MSELoss()
    criterion.to(device)

    running_loss = 0
    data_size = len(test_dataloader.dataset)
    all_tar, all_pred = torch.Tensor([[0]]).to(device), torch.Tensor([[0]]).to(device)
    all_date = torch.Tensor([0]).to(device)
    all_lat, all_lon = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)

    with torch.no_grad():
        for date, lat, lon, inputs, target in test_dataloader:
            date, lat, lon = date.to(device), lat.to(device), lon.to(device)
            inputs, target = inputs.to(device), target.to(device)
            target = target.unsqueeze(1)
            output = dbn_model(inputs)
            loss = criterion(output, target)
            running_loss += loss.item()

            all_tar = torch.cat((all_tar, target), 0)
            all_pred = torch.cat((all_pred, output), 0)

            all_date = torch.cat((all_date, date), 0)
            all_lat = torch.cat((all_lat, lat), 0)
            all_lon = torch.cat((all_lon, lon), 0)

        running_loss /= data_size
        all_date = all_date.cpu().detach().numpy().flatten()
        all_lat = all_lat.cpu().detach().numpy().flatten()
        all_lon = all_lon.cpu().detach().numpy().flatten()

        all_x = all_tar.cpu().detach().numpy().flatten()
        all_y = all_pred.cpu().detach().numpy().flatten()
        r2 = r2_score(all_x[1:], all_y[1:])

        return all_date[1:], all_lat[1:], all_lon[1:], all_x[1:], all_y[1:], r2, running_loss


def predict_train(dbn_model, optim, epochs, resume, check_path, save_path, train_dataloader, test_dataloader):
    start_epoch = -1
    if resume:
        path_checkpoint = check_path  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        dbn_model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    training = trange(start_epoch + 1, start_epoch + 1 + epochs)
    train_r2, train_loss = [], []
    test_r2, test_loss = [], []

    for epoch in training:

        t_r2, t_loss = train_one_epoch(dbn_model, optim, train_dataloader)
        train_r2.append(t_r2)
        train_loss.append(t_loss)

        v_r2, v_loss = evaluate(dbn_model, test_dataloader)
        test_r2.append(v_r2)
        test_loss.append(v_loss)

        training.set_description(
            str({'epoch': epoch + 1,
                 'train_r2': round(t_r2, 3), 'train_loss': round(t_loss, 4),
                 'val_r2': round(v_r2, 3), 'val_loss': round(v_loss, 4)}))
        if (epoch + 1) >= 1:
            checkpoint = {
                'epoch': epoch,
                'model': dbn_model.state_dict(),
                'optimizer': optim.state_dict(),
            }
            # i是用来标记交叉验证的
            torch.save(
                checkpoint,
                save_path + 'checkpoint_' + str(epoch + 1) + '.pth')
    torch.save(dbn_model, 'dbn.pth')
    d = {'train_r2': train_r2, 'train_loss': train_loss, 'val_r2': test_r2, 'val_loss': test_loss}
    df_r2_loss = pd.DataFrame(d)
    csv_path = save_path + '/' + '_' + str(start_epoch + 2) + '_' + str(start_epoch + 1 + epochs) + '_r2_loss.csv'
    df_r2_loss.to_csv(path_or_buf=csv_path, index=False)


def get_checkpoint_list(file_path, file_type, start_end_num):
    file_path = file_path
    file_list = []
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(file_type + " files not find in {}".format(file_path))
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == '.' + file_type:
                ck_num = int(os.path.splitext(file)[0].split('_')[1])
                if start_end_num[0] <= ck_num <= start_end_num[1]:
                    file_list.append([file, os.path.join(root, file)])

    return file_list


def sample_based_cv(start_end, layers, date, lat, lon, data, target, batch_size, epoch, save_path, train_mode=True):
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
    i = 1
    cv_target, cv_predict = np.array([]), np.array([])
    cv_date, cv_lat, cv_lon = np.array([]), np.array([]), np.array([])
    for train_index, test_index in cv.split(target):
        if start_end[0] <= i <= start_end[1]:

            print("cv:", i, "TRAIN:", np.shape(train_index)[0], "TEST:", np.shape(test_index)[0])
            train_con, val_con = data[train_index], data[test_index]
            train_tar, val_tar = target[train_index], target[test_index]

            train_date, val_date = date[train_index], date[test_index]
            train_lat, val_lat = lat[train_index], lat[test_index]
            train_lon, val_lon = lon[train_index], lon[test_index]

            train_dataset = MyDataset(train_date, train_lat, train_lon, train_con, train_tar)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

            val_dataset = MyDataset(val_date, val_lat, val_lon, val_con, val_tar)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

            model = dbn_pretrain(train_dataloader=train_loader,
                                 dbn_layers=layers,
                                 save_path=save_path + '/checkpoints/' + str(i) + '_china_100.pt',
                                 load_save=True)

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            if train_mode:
                Epochs = epoch
                checkpoint_path = './checkpoints/sample_cv/cv_' + str(i) + '/checkpoint_250.pth'
                # 参数 year 用来确定保存的文件夹
                train(
                    i, 'sample_cv',
                    model, optimizer, Epochs,
                    False, checkpoint_path, save_path,
                    train_dataloader=train_loader, val_dataloader=val_loader)
            else:
                ck_folder = save_path + '/checkpoints/sample_cv/cv_' + str(i)
                file_list = get_checkpoint_list(ck_folder, 'pth', [180, 200])
                ck_path = ''
                max_r2 = 0
                date2, lat2, lon2 = np.array([]), np.array([]), np.array([])
                x2, y2, rr2, ll2 = np.array([]), np.array([]), np.array([]), np.array([])
                for ck_n, ck_p in tqdm(file_list):
                    date1, lat1, lon1, x1, y1, rr, ll = test(model, ck_p, val_loader)
                    if rr > max_r2:
                        max_r2 = rr
                        ck_path = ck_p
                        date2, lat2, lon2, x2, y2, rr2, ll2 = date1, lat1, lon1, x1, y1, rr, ll

                cv_target = np.concatenate((cv_target, x2))
                cv_predict = np.concatenate((cv_predict, y2))
                print("cp_path: ", ck_path, "val_r2: ", str(rr2))

                cv_date = np.concatenate((cv_date, date2))
                cv_lat = np.concatenate((cv_lat, lat2))
                cv_lon = np.concatenate((cv_lon, lon2))

            i += 1
        else:
            i += 1
            continue

    if not train_mode:
        d = {'date': cv_date, 'lat': cv_lat, 'lon': cv_lon, 'target': cv_target, 'predict': cv_predict}
        df_x_y = pd.DataFrame(d)
        csv_path = save_path + '/r2_loss/sample_cv_target_predict_ps.csv'
        df_x_y.to_csv(path_or_buf=csv_path, index=False)


def predictive_ability_test(x_train, y_train, x_test, y_test, batch_size, epoch, save_path):
    x_train, y_train, x_test, y_test

    train_dataset = MyDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = MyDataset(x_test, y_test)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = dbn_pretrain(train_dataloader=train_loader,
                         dbn_layers=dbn_layers,
                         save_path=save_path + '/' + 'GFALL_PM2.5_自相关.pt',
                         load_save=False)


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    Epochs = epoch
    checkpoint_path = '.checkpoint1/checkpoint_200.pth'
    predict_train(
         model, optimizer, Epochs,
        False, checkpoint_path, save_path,
        train_loader, val_loader)


def get_cv_result(r2_loss_p, start_row, check_p):
    csv_list = get_file_list(r2_loss_p, 'csv')
    checkpoint_list = []
    for csv_name, csv_path in csv_list:
        cv_num = csv_name.split('_')[0]
        r2_loss_df = pd.read_csv(csv_path)
        last_50 = r2_loss_df.iloc[start_row:, :]  # 只取最后50行
        max_idx = last_50['val_r2'].idxmax()
        max_idx += 1
        check_f = check_p + '/cv_' + cv_num + '/checkpoint_' + str(max_idx) + '.pth'
        checkpoint_list.append(check_f)
    return checkpoint_list


def create_tiff(img_path, im_data, im_geotrans, im_proj):
    # gdal数据类型
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(img_path, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


def dbn_predict(pre_train_p, check_p, test_arr, tif_x, tif_y, standardScaler):
    print('执行了？')
    dbn = DBN(9, dbn_layers, savefile=pre_train_p)
    model = dbn.load_pretrained_model()  # dbn已经过训练, 直接将 savefile 加载到model

    model.to(device)
    checkpoint = torch.load(check_p)
    model.load_state_dict(checkpoint['model'])
    output_img = np.zeros([tif_y, tif_x])
    for i in tqdm(range(0, tif_y)):
        x_row = test_arr[i, :, :]
        z_index = np.where(np.isnan(x_row))
        zero_index = np.unique(z_index[0])  # 提取出含有nan的列的索引
        if zero_index.shape[0] == tif_x:
            continue

        all_index = np.arange(0, tif_x)
        non_zero_index = np.setdiff1d(all_index, zero_index)  # 去掉有nan的列索引, 保留其余索引
        input_array = x_row[non_zero_index]
        input_array = standardScaler.transform(input_array)

        input_array = input_array.astype('float32')

        with torch.no_grad():
            x = torch.from_numpy(input_array).to(device)
            y = model(x).cpu().detach().numpy().flatten()
            output_img[i, non_zero_index] = y


    return output_img



if __name__ == '__main__':


    x_train, y_train, x_test, y_test, standardScaler = LoadCSV(r"J:\毕业论文\站点数据提取\231007_GFALLTOA_QX_PM_自相关.csv")

    predictive_ability_test(x_train, y_train, x_test, y_test, 256, 3000, r'J:\毕业论文\Deep Learning\\')
    exit(0)
    train_scaler = getStandardScaler(r"J:\AODinversion\匹配数据\dailyAddMonth_train.csv")
    tifPaths = glob.glob(r'J:\AODinversion\AOD的TOA数据\月平均\Landsat\nber\*.tif')
    for tifPath in tifPaths:
        date = os.path.basename(tifPath).split('_')[0]
        strDate = str(date[0: 4]) + '-' + str(date[5: 7])
        tcwvPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tcwv',  strDate + '_resample.tif')
        tcoPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tco', strDate + '_resample.tif')
        demPath = r"J:\机器学习测试文件\dem_tif\nber_dem_no_resample.tif"
        if GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath) == None:
            continue
        else:
            test_arr, tif_x, tif_y, tif_projection, tif_transform = GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath)
        tif = gdal.Open(tifPath)
        tif_x, tif_y = ReadWrite_h5.get_RasterXY(tif)
        tif_arr = ReadWrite_h5.get_RasterArr(tif, tif_x, tif_y)[:4, :, :]

        result = dbn_predict(r'.\checkpoint_month\_china_100.pt', r'G:\KongTianYuan\DataOperations\DBN\Band4Model\checkpoint_266.pth', test_arr, tif_x, tif_y, train_scaler)
        result[tif_arr[0] == -1] = 0
        ReadWrite_h5.write_tif(result, tif_projection, tif_transform, os.path.join(r'J:\AODinversion\result\monthResult\Landsat\nber', strDate + '.tif'))


    train_scaler = getStandardScaler(r"J:\AODinversion\匹配数据\dailyAddMonth_train.csv")
    tifPaths = glob.glob(r'J:\AODinversion\AOD的TOA数据\月平均\Sentinel-2\nber\*.tif')
    for tifPath in tifPaths:
        date = os.path.basename(tifPath).split('_')[0]
        strDate = str(date[0: 4]) + '-' + str(date[5: 7])
        tcwvPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tcwv',  strDate + '_resample.tif')
        tcoPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tco', strDate + '_resample.tif')
        demPath = r"J:\机器学习测试文件\dem_tif\nber_dem_no_resample.tif"
        if GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath) == None:
            continue
        else:
            test_arr, tif_x, tif_y, tif_projection, tif_transform = GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath)
        tif = gdal.Open(tifPath)
        tif_x, tif_y = ReadWrite_h5.get_RasterXY(tif)
        tif_arr = ReadWrite_h5.get_RasterArr(tif, tif_x, tif_y)[:4, :, :]

        result = dbn_predict(r'.\checkpoint_month\_china_100.pt', r'G:\KongTianYuan\DataOperations\DBN\Band4Model\checkpoint_266.pth', test_arr, tif_x, tif_y, train_scaler)
        result[tif_arr[0] == -1] = 0
        ReadWrite_h5.write_tif(result, tif_projection, tif_transform, os.path.join(r'J:\AODinversion\result\monthResult\sentinel-2\nber', strDate + '.tif'))


    train_scaler = getStandardScaler(r"J:\AODinversion\匹配数据\dailyAddMonth_train.csv")
    tifPaths = glob.glob(r'J:\四国20年影像1\月平均\GF-1\尼泊尔\*.tif')
    for tifPath in tifPaths:
        date = os.path.basename(tifPath).split('_')[0]
        strDate = str(date[0: 4]) + '-' + str(date[5: 7])
        tcwvPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tcwv',  strDate + '_resample.tif')
        tcoPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tco', strDate + '_resample.tif')
        demPath = r"J:\机器学习测试文件\dem_tif\nber_dem_no_resample.tif"
        if GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath) == None:
            continue
        else:
            test_arr, tif_x, tif_y, tif_projection, tif_transform = GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath)
        tif = gdal.Open(tifPath)
        tif_x, tif_y = ReadWrite_h5.get_RasterXY(tif)
        tif_arr = ReadWrite_h5.get_RasterArr(tif, tif_x, tif_y)

        result = dbn_predict(r'.\checkpoint_month\_china_100.pt', r'G:\KongTianYuan\DataOperations\DBN\Band4Model\checkpoint_266.pth', test_arr, tif_x, tif_y, train_scaler)
        result[tif_arr[0] == -1] = 0
        ReadWrite_h5.write_tif(result, tif_projection, tif_transform, os.path.join(r'J:\AODinversion\result\monthResult\GF1\nber', strDate + '.tif'))


    train_scaler = getStandardScaler(r"J:\AODinversion\匹配数据\dailyAddMonth_train.csv")
    tifPaths = glob.glob(r'J:\四国20年影像1\月平均\GF-6\尼泊尔\*.tif')
    for tifPath in tifPaths:
        date = os.path.basename(tifPath).split('_')[0]
        strDate = str(date[0: 4]) + '-' + str(date[5: 7])
        tcwvPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tcwv', strDate + '_resample.tif')
        tcoPath = os.path.join(r'J:\AODinversion\辅助数据\月平均数据重采样\nber\tco', strDate + '_resample.tif')
        demPath = r"J:\机器学习测试文件\dem_tif\nber_dem_no_resample.tif"
        if GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath) == None:
            continue
        else:
            test_arr, tif_x, tif_y, tif_projection, tif_transform = GetImageData.GetAODImgData(tifPath, tcwvPath,
                                                                                               tcoPath, demPath)
        tif = gdal.Open(tifPath)
        tif_x, tif_y = ReadWrite_h5.get_RasterXY(tif)
        tif_arr = ReadWrite_h5.get_RasterArr(tif, tif_x, tif_y)

        result = dbn_predict(r'.\checkpoint_month\_china_100.pt',
                             r'G:\KongTianYuan\DataOperations\DBN\Band4Model\checkpoint_266.pth', test_arr, tif_x,
                             tif_y, train_scaler)
        result[tif_arr[0] == -1] = 0
        ReadWrite_h5.write_tif(result, tif_projection, tif_transform,
                               os.path.join(r'J:\AODinversion\result\monthResult\GF6\nber', strDate + '.tif'))


    # tifPaths = glob.glob(r'J:\AODinversion\AOD的TOA数据\sentinel-2_0.001\*.tif')
    # for tifPath in tifPaths:
    #     date = os.path.basename(tifPath).split('_')[0]
    #     strDate = str(date[0: 4]) + '-' + str(date[4: 6] + '-' + str(date[6: 8]))
    #     tcwvPath = os.path.join(r'J:\AODinversion\辅助数据\筛选数据补\resample\nber\tcwv',  strDate + '_tcwv_resample.tif')
    #     tcoPath = os.path.join(r'J:\AODinversion\辅助数据\筛选数据补\resample\nber\tco', strDate + '_tco3_resample.tif')
    #     demPath = r"J:\机器学习测试文件\dem_tif\nber_dem_no_resample.tif"
    #     if GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath) == None:
    #         continue
    #     else:
    #         test_arr, tif_x, tif_y, tif_projection, tif_transform = GetImageData.GetAODImgData(tifPath, tcwvPath, tcoPath, demPath)
    #     tif = gdal.Open(tifPath)
    #     tif_x, tif_y = ReadWrite_h5.get_RasterXY(tif)
    #     tif_arr = ReadWrite_h5.get_RasterArr(tif, tif_x, tif_y)
    #
    #     result = dbn_predict(r'.\checkpoint1\_china_100.pt', r'G:\KongTianYuan\DataOperations\DBN\checkpoint1\checkpoint_300.pth', test_arr, tif_x, tif_y, train_scaler)
    #     result[tif_arr[0] == -1] = 0
    #     ReadWrite_h5.write_tif(result, tif_projection, tif_transform, os.path.join(r'J:\AODinversion\result\Sentinel-2_result', strDate + '.tif'))


