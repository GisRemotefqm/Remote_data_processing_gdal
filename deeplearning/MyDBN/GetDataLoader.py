# --coding:utf-8--
import torch
torch.manual_seed(1)
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):

    def __init__(self, csvPath, mode='train'):
        super(MyDataset, self).__init__()
        df = pd.read_csv(csvPath, encoding='gbk')
        df = df[['Band1', 'Band2', "Band3", 'Band4', 'SZA', 'SAA', 'VZA', 'VAA']]
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)
        self.x_target = None
        self.y_target = None
        if mode == 'train':
            x_train = torch.FloatTensor(x_train)
            y_train = torch.FloatTensor(y_train)
            self.x_target, self.y_target = x_train, y_train
        else:
            x_test = torch.FloatTensor(x_test)
            y_test = torch.FloatTensor(y_test)
            self.x_target, self.y_target = x_test, y_test


    def __len__(self):

        return len(self.x_target)


    def __getitem__(self, idx):

        return self.x_target[idx], self.y_target[idx]
