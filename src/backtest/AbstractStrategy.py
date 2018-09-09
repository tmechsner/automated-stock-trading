from predictors.AbstractPredictor import AbstractPredictor


class AbstractStrategy:
    """
    Base class for Trading Strategies.
    """
    def __init__(self):
        pass

    def persist(self):
        raise NotImplementedError()

    def backtest_step(self, X, y, todays_closes, todays_date: str, accumulated_splits, todays_features,
                      training_years, trading_days_per_year) -> int:
        """
        Predict whether the stock price will rise or fall.
        :param X: All stocks' features of past days until 'today'
        :param y: All stocks' prediction targets of past days until 'today'
        :param todays_closes: All stocks' close prices of 'today'
        :param todays_date: Date of today as string
        :param accumulated_splits: All stocks' split rates since the last trading day
        :param todays_features: Features of 'today'
        :param training_years: How many years of buffer time without trading for training batch models.
        :param trading_days_per_year: How many trading days has a year
        :return: Prediction (rise / fall)
        """
        raise NotImplementedError()
