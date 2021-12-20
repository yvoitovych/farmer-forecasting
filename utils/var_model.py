import logging
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

class VanillaVAR:
    def __init__(self, data_train : pd.DataFrame, data_test : pd.DataFrame) -> None:
        logging.basicConfig(level=logging.INFO)
        self.__data_train =  data_train
        self.__data_test = data_test
        self.__model = VAR(self.__data_train)
        logging.info("Vanilla VAR initialized")

    def _select_lag_order(self, maxlags : int = 5) -> int:
        logging.info('Selecting lag order')
        p = self.__model.select_order(maxlags=maxlags)
        logging.info('Order selected')
        return p

    def train(self, lag_order: int = 2):
        logging.info("Start fitting")
        self.__model.fit(lag_order)
        logging.info("Fitted succesfully")

    def predict(self):
        pass

    def validate(self):
        pass


class SparseVAR(VanillaVAR):    
    def __init__(self) -> None:
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def validate(self):
        super().validate()