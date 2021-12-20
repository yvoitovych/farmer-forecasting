import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, accuracy_score

class VanillaVAR:
    def __init__(self, data_train : pd.DataFrame, data_test : pd.DataFrame) -> None:
        logging.basicConfig(level=logging.INFO)
        self.__data_train =  data_train
        self.__data_test = data_test
        self.__all_data = np.concatenate((self.__data_train, self.__data_test), axis=0)
        self.__model = VAR(self.__data_train)
        self.__lag_order = 0
        logging.info("Vanilla VAR initialized")

    def _select_lag_order(self, maxlags : int = 5) -> int:
        logging.info('Selecting lag order')
        p = self.__model.select_order(maxlags=maxlags)
        logging.info('Order selected')
        return p

    def train(self, lag_order: int = None):
        logging.info("Start fitting")
        if lag_order is None:
            res = self._select_lag_order(maxlags=15)
        else:
            p =lag_order
        self.__model = self.__model.fit(res.selected_orders['aic'])
        self.__lag_order = self.__model.k_ar
        logging.info("Fitted succesfully")

    def predict(self, input_data):
        pred = self.__model.forecast(y=input_data, steps=1)
        return pred

    @staticmethod
    def compare_signs(pred, real):
        print(np.sign(real))
        print(np.sign(pred))
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
        sgn = VanillaVAR.compare_signs(preds, reals)
        logging.info(f'Compare signs: {sgn}')



class SparseVAR(VanillaVAR):    
    def __init__(self) -> None:
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def validate(self):
        super().validate()