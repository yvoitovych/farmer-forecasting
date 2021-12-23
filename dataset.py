import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
from torch import nn


class TradeDataset(torch.utils.data.Dataset):
    PAD_TOKEN = -1
    def __init__(self, data_config, train):
        self.seq_length = data_config['seq_length']
        self.path = data_config['path']
        # self.point_dfs = [pd.read_csv(path) for path in self.pathes]
        self.trade_df = pd.read_csv(self.path)
        if train:
            self.trade_df = self.trade_df.iloc[:int(0.8*self.trade_df.shape[0])]
        if not train:
            self.trade_df = self.trade_df.iloc[int(0.8 * self.trade_df.shape[0]):]
        self.price_indices = list(range(0, 10)) + list(range(20, 30))
        self.amount_indices = list(range(10, 20)) + list(range(30, 40))

        # for i, df in enumerate(self.point_dfs):
        #     df['ds_num'] = i
        # self.point_df = pd.concat(self.point_dfs, ignore_index=True)
        # self.point_df['pid'] = pd.factorize(pd._libs.lib.fast_zip([self.point_df.point_id.values, self.point_df.ds_num.values]))[0]

        # self.point_df.set_index(['pid', 'order'], inplace=True)


    def __len__(self):
        return self.trade_df.shape[0] - self.seq_length - 1

    def __getitem__(self, item):
        tx = self.trade_df.iloc[item:item+self.seq_length].values
        ty = self.trade_df.iloc[item+self.seq_length:item+self.seq_length+1].values
        vwap = np.dot(ty[:, self.price_indices], ty[:, self.amount_indices].T) /  ty[:, self.amount_indices].sum()

        return tx, vwap





        # label = row_df.label.any()
        # data = row_df.iloc[:200][['x', 'y']].values

        # input = np.full((200, 2), TrajectoryDataset.PAD_TOKEN)
        # input[:data.shape[0]] = data
        # return input, label
