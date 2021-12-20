import logging
from matplotlib.pyplot import sca
import pandas as pd
import numpy as np
from typing import Tuple, List
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LOBPreprocessor:
    VWAP = 'VWAP'

    def __init__(self, path:str='data/lob_data.csv') -> None:
        logging.basicConfig(level=logging.INFO)
        self.__df_data = pd.read_csv(path, dtype=np.float64)
        self.__len = len(self.__df_data)
        
        self.__price_cols =  {
            'bid': [col for col in self.__df_data.columns if 'bids_price' in col],
            'ask': [col for col in self.__df_data.columns if 'asks_price' in col]                    
            }
        self.__vol_cols = {
            'bid': [col for col in self.__df_data.columns if 'bids_amount' in col],
            'ask': [col for col in self.__df_data.columns if 'asks_amount' in col]
            }
        self.__window_size = len(self.__vol_cols['bid'])
        self.__diff_data = None
        self.__vwap = None
        logging.info('Preprocessor initialized!')

        
    @property
    def lob_data(self) -> pd.DataFrame:
        return self.__df_data

    @property
    def diff_data(self) -> pd.DataFrame:
        return self.__diff_data

    @property
    def vwap(self) -> pd.Series:
        return self.__vwap
    
    @property
    def columns(self) -> List[str]:
        return self.__df_data.columns

    def diff_time_series(self, order:int=1) -> None:
        if self.__diff_data is None:
            self.__diff_data = deepcopy(self.__df_data)
            for _ in range(order):
                temp_df = pd.DataFrame()
                for col in self.columns:
                    temp_df[col] = np.diff(self.__diff_data[col])
                self.__diff_data = temp_df
            logging.info('Data differentiated')
        else:
            logging.warning('Data is differentiated already')

    def __calc_vwap(self) -> pd.Series:
        def __row_vwap(row):
            row_vwap = 0
            for key in ('bid', 'ask'):
                row_vwap += sum(row[self.__vol_cols[key]].to_numpy() * row[self.__price_cols[key]].to_numpy())
            return row_vwap

        volume = self.__df_data[self.__vol_cols['bid']].sum(axis=1) + self.__df_data[self.__vol_cols['ask']].sum(axis=1)
        self.__vwap = self.__df_data.apply(__row_vwap, axis=1).div(volume)

    def add_vwap(self) -> None:
        if self.__vwap is None:
            self.__calc_vwap()
            logging.info('VWAP added!')
        else:
            logging.warning('VWAP column exists!')

    def price_only(self) -> pd.DataFrame:
        return pd.concat((self.__diff_data[self.__price_cols['bid']], self.__diff_data[self.__price_cols['ask']]) , axis=1)

    def amount_only(self) -> pd.DataFrame:
        return pd.concat((self.__diff_data[self.__vol_cols['bid']] , self.__diff_data[self.__vol_cols['ask']]), axis=1 )
        
    @staticmethod
    def train_test_split(data, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_len = len(data)
        sep_idx = int(train_frac*data_len)
        return data[:sep_idx], data[sep_idx:]
  
    @staticmethod
    def normalize(time_series_data : pd.DataFrame, scaler : StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
        logging.info('Starting normalize data')
        if scaler is None:
            scaler = StandardScaler()
        normalized_time_series = scaler.fit_transform(time_series_data)
        logging.info('Data is normalized')
        return normalized_time_series, scaler
    

    # def run(self) -> None:
    #     self.add_vwap()
    #     self.diff_time_series(1)