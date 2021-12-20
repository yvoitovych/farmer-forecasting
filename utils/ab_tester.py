import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECMResults, coint_johansen
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.stattools import grangercausalitytests

class ABTester:
    def __init__(self, lob_df: pd.DataFrame) -> None:
        self.__lob_df = lob_df

    def test_granger(self):
        pass

    def test_impulse_response(self):
        pass

    def test_cointegration(self):
        pass

    def test_instanteneous_causality(self):
        pass

    def test_adf(self):
        pass

    def test_kpss(self):
        pass

