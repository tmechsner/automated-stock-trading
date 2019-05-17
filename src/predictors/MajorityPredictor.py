import numpy as np

from predictors.AbstractPredictor import AbstractPredictor


class MajorityPredictor(AbstractPredictor):
    def __init__(self):
        super(MajorityPredictor, self).__init__()
        pass

    def predict(self, training_data, training_labels, todays_features, training_years, trading_days_per_year) -> int:
        return np.argmax(np.bincount(training_labels[-1 * training_years * trading_days_per_year:]))

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        return []