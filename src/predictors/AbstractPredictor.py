import pandas as pd


class AbstractPredictor:
    """
    Base class for Stock Predictors.
    """
    def __init__(self):
        pass

    def predict(self, training_data, training_labels, todays_features, training_years, trading_days_per_year) -> int:
        """
        Predict the trend of a single stock based on knowledge about the past.
        :param training_data: Features of all days until today.
        :param training_labels: Real development of stock price of all days until today -> prediction targets for training.
        :param todays_features: Features of today, to be used for actual prediction.
        :param training_years: How many days to use for training in batch algorithms. There will be a buffer
        :param trading_days_per_year:
        :return:
        """
        raise NotImplementedError()

    def tune(self, stock_data, symbols, num_features, measure, trading_frequency, training_years, trading_days_per_year):
        """
        Tune this predictor with real stock data. Returns an array of optimal hyperparameters that can be used
        to instantiate a new predictor.
        :param stock_data:
        :param symbols:
        :param num_features:
        :param measure: Performance measure to optimize: 'f1' or 'accuracy'
        :param trading_frequency:
        :param training_years:
        :param trading_days_per_year:
        :return: an array of optimal hyperparameters that can be used to instantiate a new predictor.
        """
        raise NotImplementedError()
