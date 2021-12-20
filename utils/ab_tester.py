import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from statsmodels.tsa.vector_ar.vecm import VECMResults, coint_johansen
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.stattools import grangercausalitytests, kpss, adfuller

class ABTester:
    def __init__(self, lob_npy: np.ndarray, columns: List[str]) -> None:
        logging.basicConfig(level=logging.INFO)
        self.__lob_npy = lob_npy
        self.__columns = columns
        logging.info('A/B tester initialized')

    def test_granger(self) -> Tuple[str, bool]:
        pass

    def test_impulse_response(self) -> Tuple[str, bool]:
        pass

    def test_cointegration(self) -> Tuple[str, bool]:
        pass

    def test_instanteneous_causality(self) -> Tuple[str, bool]:
        pass

    def test_adf(self) -> Tuple[Tuple[str, bool]]:
        logging.info('Begin Augmented Dickey–Fuller test')
        for idx, col in enumerate(self.__columns):
            print(adfuller(self.__lob_npy[:, idx]))
        logging.info('Augmented Dickey–Fuller test is finished')
        return (('', 0), )

    def test_kpss(self) -> Tuple[Tuple[str, bool]]:
        logging.info('Begin KPSS test')
        for idx, col in enumerate(self.__columns):
            print(kpss(self.__lob_npy[:, idx]))
        logging.info('KPSS test is finished')
        return (('', 0), )

    # def run_tests(self) -> None:
    #     logging.info('Starting A/B tests.')
    #     logging.info('Checking stationarity.')
    #     res_adf  = self.test_adf()
    #     res_kpss  = self.test_kpss()
    #     logging.info('Performing Granger test.')
    #     res_granger, _ = self.test_granger()
    #     logging.info('Performing Cointegration test.')
    #     res_coint, _ = self.test_cointegration()
    #     logging.info('Checking instanteneous causality.')
    #     res_ic, _ = self.test_instanteneous_causality()
    #     logging.info('Checking impulse response.')
    #     res_ir, _ = self.test_impulse_response()



        



        

    