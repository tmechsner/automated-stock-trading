import os
import pickle

from backtest.AbstractStrategy import AbstractStrategy

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics


class SingleValueStrategy(AbstractStrategy):
    def __init__(self, predictor_class, predictor_params: list, name: str, init_cash_balance: float, buy_signal: int,
                 sell_signal: int, keep_signal: int, trans_cost_fixed: float, trans_cost_percentual: float,
                 min_trans: int, data_path: str, force_recalculate=False):
        """
        Initiate trading strategy.
        :param predictor_class: Predictor to use for deciding which stock to buy.
        :param predictor_params: Parameters used to initialize the predictor.
        :param name: Name of this instance of strategy-predictor combination.
        :param init_cash_balance: Initial cash balance available for trading.
        :param buy_signal: Class label for rising stocks to be found in training data.
        :param sell_signal: Class label for falling stocks to be found in training data.
        :param force_recalculate: Don't use persisted results of earlier runs.
        """
        self.predictor_class = predictor_class
        self.predictor_params = predictor_params
        self.name = name
        self.init_cash_balance = init_cash_balance
        self.trans_cost_fixed = trans_cost_fixed
        self.trans_cost_percentual = trans_cost_percentual
        self.min_trans = min_trans
        self.data_path = data_path
        self.force_recalculate = force_recalculate

        self._recalculate = True

        self._BUY_SIGNAL = buy_signal
        self._SELL_SIGNAL = sell_signal
        self._KEEP_SIGNAL = keep_signal

        self._backtests = {}  # Backtest per stock
        self._predictors = {}  # Predictors per stock
        self._predictions = {}  # Predictions per stock

    def persist(self):
        for (stock, predictions) in self._predictions.items():
            data_path = os.path.join(self.data_path, '{}_{}.pkl'.format(self.name, stock))
            with open(data_path, 'wb') as file:
                pickle.dump(predictions, file)

    def backtest_step(self, X, y, todays_closes, todays_date, accumulated_splits, todays_features, training_years, trading_days_per_year):

        for stock in X.index.levels[0]:

            # print('Todays Features: ')
            # print(todays_features[stock])
            # print('Last Training day: ')
            # print(X.loc[stock].iloc[-1])

            if stock not in self._predictors.keys():
                self._predictions[stock] = {}
                self._predictors[stock] = self.predictor_class(*self.predictor_params)
                self._backtests[stock] = SingleStockBacktest(self.init_cash_balance, self.trans_cost_fixed,
                                                             self.trans_cost_percentual, self.min_trans)
                data_path = os.path.join(self.data_path, '{}_{}.pkl'.format(self.name, stock))
                if not self.force_recalculate and os.path.isfile(data_path):
                    with open(data_path, 'rb') as file:
                        self._predictions[stock] = pickle.load(file)
                        self._recalculate = False

            todays_close = todays_closes[stock]
            todays_splitrate = accumulated_splits[stock]
            # if todays_splitrate > 2 or todays_splitrate < 0.5:
            #     print('%s - %s - %f' % (stock, str(todays_date), todays_splitrate))

            backtest = self._backtests[stock]
            backtest.split(todays_splitrate)

            if self._recalculate or todays_date not in self._predictions[stock].keys():
                predictor = self._predictors[stock]
                todays_feats = list(todays_features[stock].values())

                X_train = X.loc[stock].values
                y_train = y.loc[stock].values
                signal = predictor.predict(X_train, y_train, todays_feats, training_years, trading_days_per_year)
                if type(signal) is list or type(signal) is np.ndarray:
                    signal = signal[-1]

                self._predictions[stock][todays_date] = signal
            else:
                signal = self._predictions[stock][todays_date]

            if signal == self._BUY_SIGNAL:
                backtest.buy(todays_close, todays_date)
            elif signal == self._SELL_SIGNAL:
                backtest.sell(todays_close, todays_date)
            elif signal == self._KEEP_SIGNAL:
                backtest.keep(todays_close, todays_date)

    def get_performance(self, symbols: list) -> dict:
        result = {}
        for stock in symbols:
            result[stock] = self._backtests[stock].last_balance / self.init_cash_balance
        return result

    def evaluate(self, targets: pd.DataFrame, show_probas=False):
        """
        Print min/max/mean accuracy and F1 score over all stocks and average number of buy, sell (and keep) signals.
        :param targets: DataFrame with correct labels for each stock. Stock names as index.
        :param show_probas: Plot confidence score distribution, if available.
        :return: List of f_scores (one per stock) in order of the stocks in target.index.
        """
        accuracies = []
        f_scores = []
        keep_signals = []
        buy_signals = []
        sell_signals = []
        predict_probas = []
        importances = []
        for stock in targets.index.levels[0]:
            predictions = np.array(list(self._predictions[stock].values()))
            accuracies.append(metrics.accuracy_score(targets.loc[stock].values, predictions))
            f_scores.append(metrics.f1_score(targets.loc[stock].values[predictions != self._KEEP_SIGNAL], predictions[predictions != self._KEEP_SIGNAL]))
            pred_series = pd.Series(predictions)

            try:
                importances_tmp = self._predictors[stock]._importances
                for f in range(len(importances_tmp)):
                    try:
                        importances[f] += importances_tmp[f]
                    except IndexError:
                        importances.append(importances_tmp[f])
            except AttributeError:
                pass

            if show_probas:
                try:
                    predict_probas.extend(self._predictors[stock]._predict_probas)
                except AttributeError:
                    pass
            if len(np.unique(pred_series)) > 1:
                pred_series_groupby = pred_series.groupby(pred_series).count()
                if len(pred_series_groupby) > 2:
                    keep_signals.append(pred_series_groupby.iloc[0])
                    buy_signals.append(pred_series_groupby.iloc[1])
                    sell_signals.append(pred_series_groupby.iloc[2])
                else:
                    buy_signals.append(pred_series_groupby.iloc[0])
                    sell_signals.append(pred_series_groupby.iloc[1])

        if len(importances) > 0:
            indices = np.argsort(importances)[::-1]
            imp_sum = sum(importances)

            # Print the feature ranking
            print("  Feature ranking:")
            for f in range(len(indices)):
                print("  %d. feature %d (%f)" % (f + 1, indices[f] + 1, importances[indices[f]] / imp_sum))

        if show_probas and len(predict_probas) > 0:
            plt.hist(predict_probas, bins=8)
            plt.title("  Prediction probability distribution ({})".format(self.name))
            plt.show()
        accuracies = np.array(accuracies)
        f_scores = np.array(f_scores)
        keep_signals = np.array(keep_signals)
        buy_signals = np.array(buy_signals)
        sell_signals = np.array(sell_signals)
        print('  Accuracy: min. %.2f   max. %.2f   mean %.2f' % (accuracies.min(), accuracies.max(), accuracies.mean()))
        print('  F1 Score: min. %.2f   max. %.2f   mean %.2f' % (f_scores.min(), f_scores.max(), f_scores.mean()))
        if len(buy_signals) > 0 and len(sell_signals) > 0:
            print('  Mean BUY vs SELL signals: %i - %i' % (round(buy_signals.mean(), 0), round(sell_signals.mean(), 0)))
        if len(keep_signals) > 0:
            print('  Mean keep signals: %i' % round(np.array(keep_signals).mean(), 0))
        return f_scores

    # def output_metrics(self, X_test, buy_sig_pred, buy_sig_test, sell_sig_pred, sell_sig_test,
    #                    y_test, print_class_report=True, plot_roc=False):
    #
    #     print('Buy/Sell Accuracy: %.3f' % ((buy_sig_test == buy_sig_pred).sum() / buy_sig_test.size))
    #
    #     if print_class_report:
    #         print(metrics.classification_report(buy_sig_test, buy_sig_pred, target_names=['S/H', 'Buy']))
    #         print(metrics.classification_report(sell_sig_test, sell_sig_pred, target_names=['B/H', 'Sell']))
    #
    #     if plot_roc:
    #         buy_fpr, buy_tpr, _ = metrics.roc_curve(buy_sig_test, (self.predictor.predict_proba(X_test)[:, 1]))
    #         buy_auc = metrics.auc(buy_fpr, buy_tpr)
    #         sell_fpr, sell_tpr, _ = metrics.roc_curve(sell_sig_test, (self.predictor.predict_proba(X_test)[:, 1]))
    #         sell_auc = metrics.auc(sell_fpr, sell_tpr)
    #         plt.figure()
    #         line_width = 2
    #         plt.plot(buy_fpr, buy_tpr, color='darkorange',
    #                  lw=line_width, label='Buy signal ROC curve (area = %0.4f)' % buy_auc)
    #         plt.plot(sell_fpr, sell_tpr, color='blue',
    #                  lw=line_width, label='Sell signal ROC curve (area = %0.4f)' % sell_auc)
    #         plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
    #         plt.xlim([0.0, 1.0])
    #         plt.ylim([0.0, 1.05])
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.legend(loc="lower right")
    #         plt.show()


class SingleStockBacktest:
    def __init__(self, init_cash_balance, trans_cost_fixed, trans_cost_percentual, min_trans):
        self.trans_cost_fixed = trans_cost_fixed
        self.trans_cost_percentual = trans_cost_percentual
        self.min_trans = min_trans
        self.last_balance = 0.0
        self._balances = []
        self._balances_series = pd.Series([])
        self._cash_balance = init_cash_balance
        self._num_shares = 0
        self._num_trans = 0

    def split(self, split_rate):
        self._num_shares *= split_rate

    def buy(self, price, date):
        trans_cost = (self.trans_cost_fixed + self._cash_balance * self.trans_cost_percentual) / (1 + self.trans_cost_percentual)  # trans_cost_fixed + (close_ * shares_to_buy * trans_cost_percentual)
        shares_to_buy = (self._cash_balance - trans_cost) // price
        if self._cash_balance > 0 and shares_to_buy >= self.min_trans:
            self._num_shares += shares_to_buy
            self._cash_balance -= shares_to_buy * price
            self._cash_balance -= trans_cost
            self._num_trans += 1
        self._update_balance(price, date)

    def sell(self, price, date):
        if self._num_shares > 0:
            self._cash_balance += price * self._num_shares
            self._cash_balance -= self.trans_cost_fixed
            self._cash_balance -= (price * self._num_shares * self.trans_cost_percentual)
            self._num_shares = 0
            self._num_trans += 1
        self._update_balance(price, date)

    def keep(self, price, date):
        self._update_balance(price, date)

    def _update_balance(self, last_price, date):
        balance = self._cash_balance + self._num_shares * last_price
        self._balances.append(balance)
        self._balances_series = self._balances_series.append(pd.Series(balance, index=[date]))
        self.last_balance = balance

    def get_balances_series(self):
        self._balances_series.index = pd.to_datetime(self._balances_series.index, format='%Y-%m-%d')
        return self._balances_series
