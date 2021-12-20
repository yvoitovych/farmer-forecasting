import logging
import pandas as pd
import numpy as np
from typing import Tuple
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

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
        logging.info('Preprocessor initialized!')
        
    @property
    def lob_data(self):
        return self.__df_data 

    def __all_cols(self):
        return self.__df_data.columns

    def diff_time_series(self, order:int=1) -> pd.DataFrame:
        diff_data = deepcopy(self.__df_data)
        for _ in range(order):
            temp_df = pd.DataFrame()
            for col in self.__all_cols():
                temp_df[col] = np.diff(diff_data[col])
            diff_data = temp_df
        return diff_data

    def __calc_vwap(self) -> pd.Series:
        def __row_vwap(row):
            row_vwap = 0
            for key in ('bid', 'ask'):
                
                row_vwap += sum(row[self.__vol_cols[key]].to_numpy() * row[self.__price_cols[key]].to_numpy())
                
            return row_vwap

        volume = self.__df_data[self.__vol_cols['bid']].sum(axis=1) + self.__df_data[self.__vol_cols['ask']].sum(axis=1)
        
        vwap = self.__df_data.apply(__row_vwap, axis=1).div(volume)
        return vwap 


    def add_vwap(self) -> None:
        if LOBPreprocessor.VWAP not in self.__df_data.columns:
            self.__df_data[LOBPreprocessor.VWAP] = self.__calc_vwap()
            logging.info('VWAP added!')
        else:
            logging.warning('VWAP column exists!')

        

    def train_test_split(self, train_frac: float =0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sep_idx = int(train_frac*self.__len)
        return self.__df_data[:sep_idx], self.__df_data[sep_idx:]
  
    @staticmethod
    def normalize(time_series_data : pd.DataFrame):
        scaler = StandardScaler()
        normalized_time_series = scaler.fit_transform(time_series_data)
        return normalized_time_series, scaler
    
    def run(self) -> None:
        self.add_vwap()
        self.diff_time_series(1)