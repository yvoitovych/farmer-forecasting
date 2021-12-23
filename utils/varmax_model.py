import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, accuracy_score

class VanillaVARMAX:
    def __init__(self, data_train : pd.DataFrame, data_test : pd.DataFrame, ar_lag=6, ma_lag=2) -> None:
        logging.basicConfig(level=logging.INFO)
        self.__data_train =  data_train
        self.__data_test = data_test
        self.__all_data = np.concatenate((self.__data_train, self.__data_test), axis=0)

        self.__model = VARMAX(self.__data_train, order=(ar_lag, ma_lag))

        self.__lag_order = max(ar_lag, ma_lag)

        logging.info("Vanilla VAR initialized")

    
    def train(self, lag_order: int = None, maxiter: int = 100):
        logging.info("Start fitting")
        
        self.__model = self.__model.fit(maxiter=maxiter, disp=True)
        logging.info(self.__model)
        logging.info("Fitted succesfully")

    def predict(self, input_data):
        pred = self.__model.forecast(y=input_data, steps=1)
        return pred

    @staticmethod
    def compare_signs(pred, real):
        
        return np.sum(np.sign(pred)==np.sign(real))/len(real)

    def validate(self, col_id):
        test_len = len(self.__data_test)
        preds = []
        reals = []
        for i in range(test_len):
            end_idx = -test_len+i
            start_idx = end_idx - self.__lag_order
            input_data = self.__all_data[start_idx:end_idx]
            pred = self.predict(input_data=input_data)
            real = self.__all_data[end_idx]
            preds.append(pred.reshape(-1,)[col_id])
            reals.append(real.reshape(-1,)[col_id])
        
        logging.info(f'p={self.__lag_order}')
        logging.info('MAE: {0}'.format(mean_absolute_error(reals, preds))) 
        logging.info('MAPE: {0}'.format(mean_absolute_percentage_error(reals, preds)))
        logging.info('RMSE: {0}'.format(np.sqrt(mean_squared_error(reals, preds))))       


    def validate_inverse_transform(self, col_id, scaler):
        test_len = len(self.__data_test)
        preds = []
        reals = []
        for i in range(test_len):
            end_idx = -test_len+i
            start_idx = end_idx - self.__lag_order
            input_data = self.__all_data[start_idx:end_idx]
            pred = self.predict(input_data=input_data)
            real = self.__all_data[end_idx]
         
            preds.append(scaler.inverse_transform(pred).reshape(-1,)[col_id])
            reals.append(scaler.inverse_transform(real.reshape(1,-1)).reshape(-1,)[col_id])
            
        logging.info(f'p={self.__lag_order}')
        logging.info('MAE: {0}'.format(mean_absolute_error(reals, preds))) 
        logging.info('MAPE: {0}'.format(mean_absolute_percentage_error(reals, preds)))
        logging.info('RMSE: {0}'.format(np.sqrt(mean_squared_error(reals, preds)))) 

    def validate_classify(self, col_id, scaler):
        test_len = len(self.__data_test)
        preds = []
        reals = []
        for i in range(test_len):
            end_idx = -test_len+i
            start_idx = end_idx - self.__lag_order
            input_data = self.__all_data[start_idx:end_idx]
            pred = self.predict(input_data=input_data)
            real = self.__all_data[end_idx]
         
            preds.append(scaler.inverse_transform(pred).reshape(-1,)[col_id])
            reals.append(scaler.inverse_transform(real.reshape(1,-1)).reshape(-1,)[col_id])
           
        logging.info(f'p={self.__lag_order}')
        sgn = VanillaVARMAX.compare_signs(preds, reals)
        logging.info(f'Compare signs: {sgn}')
