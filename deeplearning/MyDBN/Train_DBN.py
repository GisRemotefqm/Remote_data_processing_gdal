# --coding:utf-8--
import random
from DBN import DBN
import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random

random.seed(1)
torch.cuda.device_count()
from torch.utils.data import TensorDataset, DataLoader, Dataset

standardScaler = StandardScaler()

def LoadCSV(csvPath):

    df = pd.read_csv(csvPath, encoding='gbk');
    df = df[['B2', 'B3', 'B4', 'B5', 'tcwv', 'tco', 'EVI',
             'Site_Elevation(m)', 'AOD_550nm']];
    # , 'tcwv', 'tco', 'EVI'
    X = df.iloc[:, :-1].to_numpy();
    Y = df.iloc[:, -1].to_numpy();

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)

    y_train1 = y_train[:, np.newaxis]
    print(np.hstack((x_train, y_train1)).shape)
    pd.DataFrame(np.hstack((x_train, y_train1)), columns=['B2', 'B3', 'B4', 'B5', 'tcwv', 'tco', 'EVI',
                                                                   'Site_Elevation(m)',
                                                                   'AOD_550nm']).to_csv(
        r'J:\AODinversion\匹配数据\MydailyAddMonth_train.csv', index=False)
    y_test1 = y_test[:, np.newaxis]
    pd.DataFrame(np.hstack((x_test, y_test1)), columns=['B2', 'B3', 'B4', 'B5', 'tcwv', 'tco', 'EVI',
                                                                'Site_Elevation(m)',
                                                                'AOD_550nm']).to_csv(
        r'J:\AODinversion\匹配数据\MydailyAddMonth_test.csv', index=False)

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


def sampleCVLoadCSV(csvPath):

    df = pd.read_csv(csvPath, encoding='gbk');
    df2 = df[['band3', 'band4']]
    df = df[['blh', 'r', 'sp', 't2m', 'wd', 'ws', 'band1', 'band2', 'DEM', 'PM2.5']];

    # , 'tcwv', 'tco', 'EVI'
    X = df.iloc[:, :-1];
    ndvi = (df2['band4'].to_numpy() - df2['band3'].to_numpy()) / (df2['band4'].to_numpy() + df2['band3'].to_numpy())
    ndvi = np.nan_to_num(ndvi, nan=0)

    X['ndvi'] = ndvi
    X = X.to_numpy()
    Y = df.iloc[:, -1].to_numpy();

    standardScaler.fit(X)
    X = standardScaler.transform(X)

    x_train = torch.from_numpy(X)
    y_train = torch.from_numpy(Y)

    x_train = x_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    return x_train, y_train,


def sampleCV(model, start_end, data, target, batch_size, epoch_pretrain, epoch_finetune, loss_function, optimizer, lr_steps):
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
    i = 1
    cv_target, cv_predict = np.array([]), np.array([])

    for train_index, test_index in cv.split(target):

        if start_end[0] <= i <= start_end[1]:
            print("cv:", i, "TRAIN:", np.shape(train_index)[0], "TEST:", np.shape(test_index)[0])
            train_con, val_con = data[train_index], data[test_index]
            train_tar, val_tar = target[train_index], target[test_index]
            print('*' * 50)
            print(train_con.shape, val_con.shape)
            model.pretrain(train_con, train_tar, epoch=epoch_pretrain, batch_size=batch_size)
            x3, y3 = model.finetune(train_con, train_tar, val_con, val_tar, epoch_finetune, batch_size, loss_function, optimizer, lr_steps, True)

            cv_target = np.concatenate((cv_target, x3))
            cv_predict = np.concatenate((cv_predict, y3))
    d = {'target': cv_target, 'predict': cv_predict}
    pd.DataFrame(d).to_csv(r'J:\毕业论文(错)\Deep Learning\csv\231009_DBN_AODResult-PM2.5.csv', index=False)

input_length = 10  # 数据变量维度，暂为21
output_length = 1
batch_size = 128  # 没啥用，先放这里

# x_train, y_train, x_test, y_test, standardScaler = LoadCSV(r"J:\AODinversion\匹配数据\TOA-AOD-qxNoResampleAddMonth.csv")
x_train, y_train = sampleCVLoadCSV(r"J:\毕业论文(错)\站点数据提取\231007_GFTOA_QX_PM.csv")
# network

hidden_units = [256, 128, 64]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device == 'cuda:0':
    assert torch.cuda.is_available() is True, "cuda isn't available"
    print('Using GPU backend.\n'
          'GPU type: {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU backend.')

epoch_pretrain = 200  # 原本为100，为运行通，先取10
epoch_finetune = 1000 # 原本为200，为运行通，先取10
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam
adam_lr = 0.0001  # finetune bp 的学习率
lr_steps = 1000

# Build model
dbn = DBN(hidden_units=hidden_units, visible_units=input_length, output_units=output_length, device=device)

# Train model
# dbn.pretrain(x_train, y_train, epoch=epoch_pretrain, batch_size=batch_size)
# dbn.finetune(x_train, y_train, x_test, y_test, epoch_finetune, batch_size, loss_function,
#              optimizer(dbn.parameters(), lr=adam_lr), lr_steps, True)

sampleCV(dbn, [1, 10], x_train, y_train, batch_size, epoch_pretrain, epoch_finetune, loss_function, optimizer(dbn.parameters(), lr=adam_lr), lr_steps)

torch.save(dbn, r'J:\毕业论文(错)Deep Learning\dbn.pth')




