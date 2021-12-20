import logging
import pandas as pd
from typing import Tuple
from statsmodels.tsa.vector_ar.vecm import VECMResults, coint_johansen
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.stattools import grangercausalitytests

class ABTester:
    def __init__(self, lob_df: pd.DataFrame) -> None:
        self.__lob_df = lob_df
        self.__lob_npy = self.__lob_df.to_numpy() 
        self.__all_cols = self.__df_data.columns
        logging.basicConfig(level=logging.INFO)

    def test_granger(self) -> Tuple[str, bool]:
        pass

    def test_impulse_response(self) -> Tuple[str, bool]:
        pass

    def test_cointegration(self) -> Tuple[str, bool]:
        pass

    def test_instanteneous_causality(self) -> Tuple[str, bool]:
        pass

    def test_adf(self) -> Tuple[Tuple[str, bool]]:
        pass

    def test_kpss(self) -> Tuple[Tuple[str, bool]]:
        pass

    def run_tests(self) -> None:
        logging.info('Starting A/B tests.')
        logging.info('Checking stationarity.')
        res_adf  = self.test_adf()
        res_kpss  = self.test_kpss()
        logging.info('Performing Granger test.')
        res_granger, _ = self.test_granger()
        logging.info('Performing Cointegration test.')
        res_coint, _ = self.test_cointegration()
        logging.info('Checking instanteneous causality.')
        res_ic, _ = self.test_instanteneous_causality()
        logging.info('Checking impulse response.')
        res_ir, _ = self.test_impulse_response()



        



        

    