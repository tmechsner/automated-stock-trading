import numpy as np
import pandas as pd

from predictors.AbstractPredictor import AbstractPredictor


class BuyAndHoldPredictor(AbstractPredictor):
    def __init__(self, buy_signal):
        super(BuyAndHoldPredictor, self).__init__()
        self._BUY_SIGNAL = buy_signal
        pass

    def predict(self, X, y, todays_features, training_years, trading_days_per_year) -> int:
        return self._BUY_SIGNAL

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        return []
