import numpy as np
import pandas as pd

from predictors.AbstractPredictor import AbstractPredictor


class ConstantPredictor(AbstractPredictor):
    def __init__(self):
        super(ConstantPredictor, self).__init__()
        pass

    def predict(self, training_data, training_labels, todays_features, training_years, trading_days_per_year) -> pd.Series:
        return training_labels[-1]  # Suggest that the trend of the last trading period repeats itself

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        return []
