import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import TechnicalIndicators
from SP500Data import SP500Data
from backtest.SingleValueStrategy import *
from predictors.AbstractPredictor import AbstractPredictor

TARGET_COL = 'target'
CLOSE_COL = 'Close'
ADJ_CLOSE_COL = 'AdjClose'
ADJ_OPEN_COL = 'AdjOpen'
ADJ_LOW_COL = 'AdjLow'
ADJ_HIGH_COL = 'AdjHigh'
ADJ_VOL_COL = 'AdjVolume'
SPLIT_COL = 'SplitRatio'


class StockTrader:
    def __init__(self, start_date, end_date, dataset='sp500_simulate', trading_frequency=1, trans_cost_fixed=5, trans_cost_percentual=0.0, min_trans=100, training_years=3, trading_days_per_year=246):
        """
        :param dataset: sp500_tune or sp500_simulate
        :param trading_frequency: A value of n means that trading will be executed on every nth simulation day
        :param trans_cost_fixed: Fixed transaction cost
        :param trans_cost_percentual: Transaction cost in percentage of transaction volume (0.05 -> 5% of trans vol)
        :param min_trans: Minimum number of shares to be bought per transaction
        :param training_years: Number of years to use as training basis -> don't trade in the first training_years years
        :param trading_days_per_year: Days per year
        :return:
        """

        self.BUY_SIGNAL = 1  # if changed: Add pos_label=self._BUY_SIGNAL to all f1_score calculations!
        self.SELL_SIGNAL = 2
        self.KEEP_SIGNAL = 0

        self._start_date = start_date
        self._end_date = end_date
        self._trading_frequency = trading_frequency
        self._trans_cost_fixed = trans_cost_fixed
        self._trans_cost_percentual = trans_cost_percentual
        self._min_trans = min_trans
        self._training_years = training_years
        self._trading_days_per_year = trading_days_per_year

        self._num_features = 0

        dataset, subset = dataset.split('_')

        self._predictors = {}
        self._backtests = {}
        self._sp500 = SP500Data(self._start_date, self._end_date, tune_or_simulate=0 if subset == 'tune' else 1)
        self._symbols = self._sp500.get_components()

        self._sp500.print_stats()

        load_prepared_data = False
        two_classes_only = True

        pwd = os.path.realpath(__file__)

        self._annotated_data_path = os.path.join(os.path.dirname(pwd), 'data/{}/annotated_data'.format(dataset))
        os.makedirs(self._annotated_data_path, exist_ok=True)

        self._backtest_data_path = os.path.join(os.path.dirname(pwd), 'data/{}/backtests'.format(dataset))
        os.makedirs(self._backtest_data_path, exist_ok=True)

        print('\nUsing data from {} to {}.\nPreparing data...'.format(self._start_date, self._end_date))
        all_files_present = True
        for stock in self._symbols:
            all_files_present &= os.path.isfile(os.path.join(self._annotated_data_path, '{}.pkl'.format(stock)))

        if load_prepared_data and all_files_present:
            stock_data_complete = {}
            for stock in self._symbols:
                stock_data_complete[stock] = pd.read_pickle(os.path.join(self._annotated_data_path, '{}.pkl'.format(stock)))
        else:
            stock_data_complete = self._sp500.get()
            print('\n')
            stock_data_complete = self._build_tech_ind(stock_data_complete)
            self._num_features, stock_data_complete = self._build_features(stock_data_complete, two_features_only=two_classes_only)
            # stock_data_complete = self._normalize_data(stock_data_complete)
            symbols_to_exclude = []
            for stock in self._symbols:
                index_min = stock_data_complete[stock].index.min()
                index_max = stock_data_complete[stock].index.max()
                if index_min.date() != self._start_date or index_max.date() != self._end_date:
                    print('Excluding %s' % stock)
                    symbols_to_exclude.append(stock)
                    stock_data_complete.pop(stock)
                    continue

                stock_data_complete[stock] = self._handle_outliers(stock_data_complete[stock])
                start_index = stock_data_complete[stock].index.searchsorted(pd.Timestamp(self._start_date))
                stop_index = stock_data_complete[stock].index.searchsorted(pd.Timestamp(self._end_date))
                stock_data_complete[stock] = stock_data_complete[stock].iloc[start_index:stop_index + 1]
                stock_data_complete[stock].to_pickle(os.path.join(self._annotated_data_path, '{}.pkl'.format(stock)))
            self._symbols = list(filter(lambda symbol: symbol not in symbols_to_exclude, self._symbols))
        self._stock_data_dict = stock_data_complete
        self._stock_data_df = pd.concat(list(self._stock_data_dict.values()), keys=list(self._stock_data_dict.keys()), names=['Symbol'])
        self._num_features = len(list(filter(lambda col_name: col_name[0] == 'I' and len(col_name) <= 3, self._stock_data_df.columns.tolist())))
        print('Using %i stocks.' % len(self._symbols))
        print('Done.\n')

    @staticmethod
    def _build_tech_ind(dfs: dict) -> dict:
        """
        Build technical indicators and append them to the DataFrames as columns.
        :param dfs: Dict containing DataFrames by symbol containing all stock data as provided by group_by_stockname()
        :return: dict: {stock_name:string -> stock_data:DataFrame} with additional feature columns
        """
        result = {}
        for symbol, stock_df in dfs.items():
            pmas = TechnicalIndicators.sma(stock_df, 5, 'PMA_S', col=ADJ_CLOSE_COL)
            pmal = TechnicalIndicators.sma(stock_df, 20, 'PMA_L', col=ADJ_CLOSE_COL)
            vmas = TechnicalIndicators.sma(stock_df, 5, 'VMA_S', col=ADJ_VOL_COL)
            vmal = TechnicalIndicators.sma(stock_df, 20, 'VMA_L', col=ADJ_VOL_COL)
            rsi = TechnicalIndicators.rsi(stock_df, 14, col=ADJ_CLOSE_COL)
            sto = TechnicalIndicators.sto(stock_df, 14, 3, col=ADJ_CLOSE_COL, col_low=ADJ_LOW_COL, col_high=ADJ_HIGH_COL)
            bb = TechnicalIndicators.bbands(stock_df, 8, 2, col=ADJ_CLOSE_COL)
            result[symbol] = pd.concat([stock_df, pmas, pmal, vmas, vmal, rsi, sto, bb], axis=1)
        return result

    def _build_features(self, dfs: dict, two_features_only: bool = False) -> tuple:
        """
        Build features and append them to the DataFrames as columns.
        :param dfs: Dict containing DataFrames by symbol containing all stock data as provided by group_by_stockname()
        :return: (int, dict): number of features created, {stock_name:string -> stock_data:DataFrame} with additional feature columns
        """
        result = {}
        num_features = 0
        for symbol, stock_df in dfs.items():
            target = pd.Series(stock_df[ADJ_CLOSE_COL].diff(-self._trading_frequency), name='')
            clf_target = pd.Series(np.array([0] * len(target)), name=TARGET_COL) # dummy, will be overwritten by classification labels
            clf_target.index = target.index
            clf_target[target > 0] = self.SELL_SIGNAL   # Next day's close is smaller -> Sell
            clf_target[target < 0] = self.BUY_SIGNAL    # Next day's close is greater -> Buy
            clf_target[target.isnull()] = self.SELL_SIGNAL
            if two_features_only:
                target2 = pd.Series(stock_df[ADJ_CLOSE_COL].diff(-self._trading_frequency - 1), name='')
                clf_target2 = pd.Series(np.array([0] * len(target)), name='target2')  # dummy, will be overwritten by classification labels
                clf_target2.index = target2.index
                clf_target2[target2 >= 0] = self.SELL_SIGNAL   # Second next day's close is smaller / bigger -> Sell
                clf_target2[target2 < 0] = self.BUY_SIGNAL  # Second next day's close is greater -> Buy
                clf_target2[target2.isnull()] = self.SELL_SIGNAL
                clf_target[target == 0] = clf_target2[target == 0]  # Same price next day? Look at day after tomorrow
            else:
                clf_target[target == 0] = self.KEEP_SIGNAL

            # i1 = pd.Series(-1 + stock_df['RSI'] / 50.0, name='I1')
            # i2 = pd.Series(-1 + 2 * (stock_df[ADJ_CLOSE_COL] - stock_df['BB_L']) / (stock_df['BB_U'] - stock_df['BB_L']), name='I2')
            # # i2 = pd.Series((stock_df[ADJ_CLOSE_COL] - stock_df['BB_U']) / stock_df['BB_U'], name='I2')
            # i3 = pd.Series((stock_df[ADJ_CLOSE_COL] - stock_df['BB_L']) / stock_df['BB_L'], name='I3')
            # i4 = pd.Series(-1 + stock_df['%K'] / 50.0, name='I4')
            # i5 = pd.Series(-1 + stock_df['%D'] / 50.0, name='I5')
            # i6 = pd.Series(stock_df['%K'].diff() / 100.0, name='I6')
            # i7 = pd.Series(stock_df['%D'].diff() / 100.0, name='I7')
            # i8 = pd.Series(stock_df[ADJ_CLOSE_COL].diff() / stock_df[ADJ_CLOSE_COL].shift(1), name='I8')
            # i9 = pd.Series((stock_df[ADJ_CLOSE_COL] - stock_df[ADJ_LOW_COL]) / (stock_df[ADJ_HIGH_COL] - stock_df[ADJ_LOW_COL]), name='I9')
            # i10 = pd.Series((stock_df['PMA_S'] - stock_df['PMA_S'].shift(1)) / stock_df['PMA_S'].shift(1), name='I10')
            # i11 = pd.Series((stock_df['PMA_L'] - stock_df['PMA_L'].shift(1)) / stock_df['PMA_L'].shift(1), name='I11')
            # i12 = pd.Series((stock_df['PMA_S'] - stock_df['PMA_L'].shift(1)) / stock_df['PMA_L'].shift(1), name='I12')
            # # i12 = pd.Series((stock_df['PMA_S'] - stock_df['PMA_L']) / stock_df['PMA_L'], name='I12')
            # i13 = pd.Series((stock_df[ADJ_CLOSE_COL] - stock_df['PMA_L']) / stock_df['PMA_L'], name='I13')
            # i14 = pd.Series((stock_df[ADJ_CLOSE_COL] - stock_df[ADJ_CLOSE_COL].rolling(6).min()) / stock_df[ADJ_CLOSE_COL].rolling(6).min(), name='I14')
            # i15 = pd.Series((stock_df[ADJ_CLOSE_COL] - stock_df[ADJ_CLOSE_COL].rolling(6).max()) / stock_df[ADJ_CLOSE_COL].rolling(6).max(), name='I15')
            # i16 = pd.Series((stock_df[ADJ_VOL_COL] - stock_df[ADJ_VOL_COL].shift(1)) / stock_df[ADJ_VOL_COL].shift(1), name='I16')
            # i17 = pd.Series((stock_df['VMA_S'] - stock_df['VMA_S'].shift(1)) / stock_df['VMA_S'].shift(1), name='I17')
            # i18 = pd.Series((stock_df['VMA_L'] - stock_df['VMA_L'].shift(1)) / stock_df['VMA_L'].shift(1), name='I18')
            # i19 = pd.Series((stock_df['VMA_S'] - stock_df['VMA_L'].shift(1)) / stock_df['VMA_L'].shift(1), name='I19')
            # # i19 = pd.Series((stock_df['VMA_S'] - stock_df['VMA_L']) / stock_df['VMA_L'], name='I19')
            # i20 = pd.Series((stock_df[ADJ_VOL_COL] - stock_df['VMA_L']) / stock_df['VMA_L'], name='I20')
            # i21 = pd.Series((stock_df[ADJ_VOL_COL] - stock_df[ADJ_VOL_COL].rolling(6).min()) / stock_df[ADJ_VOL_COL].rolling(6).min(), name='I21')
            # i22 = pd.Series((stock_df[ADJ_VOL_COL] - stock_df[ADJ_VOL_COL].rolling(6).max()) / stock_df[ADJ_VOL_COL].rolling(6).max(), name='I22')
            #
            # features = [
            #     i1,
            #     i2,
            #     i3,
            #     i4,
            #     i5,
            #     i6,
            #     i7,
            #     i8,
            #     i9,
            #     i10,
            #     i11,
            #     i12,
            #     i13,
            #     i14,
            #     i15,
            #     i16,
            #     i17,
            #     i18,
            #     i19,
            #     i20,
            #     i21,
            #     i22
            # ]

            i1 = pd.Series(-1 + stock_df['RSI'] / 50.0, name='I1')
            i2 = pd.Series(-1 + stock_df['%K'] / 50.0, name='I2')
            i3 = pd.Series(-1 + stock_df['%D'] / 50.0, name='I3')
            i4 = pd.Series(np.tanh(stock_df['PMA_S'] - stock_df['PMA_L']), name='I4')
            i5 = pd.Series(np.tanh(stock_df[ADJ_CLOSE_COL] - stock_df['PMA_S']), name='I5')
            i6 = pd.Series(np.tanh((stock_df['VMA_S'] - stock_df['VMA_L']) / (stock_df['VMA_S'] - stock_df['VMA_L']).rolling(10).mean()), name='I7')
            i7 = pd.Series(np.tanh((stock_df[ADJ_VOL_COL] - stock_df['VMA_S']) / (stock_df[ADJ_VOL_COL] - stock_df['VMA_S']).rolling(3).mean()), name='I7')
            i8 = pd.Series(-1 + 2 * (stock_df[ADJ_CLOSE_COL] - stock_df['BB_L']) / (stock_df['BB_U'] - stock_df['BB_L']), name='I6')

            features = [
                i1,
                i2,
                i3,
                i4,
                i5,
                i6,
                i7,
                i8
            ]

            i = 0
            for feat in features:
                i += 1
                features[i - 1] = feat.rename('I{}'.format(i))
                features[i - 1] = features[i - 1].fillna(features[i - 1].mean())
                # features[i - 1] = (feat - feat.mean()).rename('I{}'.format(i))
            num_features = len(features)

            new_cols = [stock_df, clf_target]
            new_cols.extend(features)
            result[symbol] = pd.concat(new_cols, axis=1)
        return num_features, result

    @staticmethod
    def _handle_outliers(df: pd.DataFrame) -> dict:
        """
        Drop rows containing NaN. This value is likely to appear in early rows as some features require several days of
        historical data to be present for being calculated.
        Replace inf and -inf with 0.0.
        :param df:
        :return:
        """
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], 0.0)
        return df

    def _normalize_data(self, dfs: dict) -> dict:
        result = dfs.copy()
        for symbol, stock_df in result.items():
            for i in range(1, self._num_features + 1):
                feat = result[symbol]['I{}'.format(i)]
                if feat.max() != feat.min():
                    result[symbol]['I{}'.format(i)] = (feat - feat.min()) / (feat.max() - feat.min())
                else:
                    result[symbol]['I{}'.format(i)] = 0.0
        return result

    def print_feature_stats(self):
        fig, ax = plt.subplots()

        print('ft\t\tmean\t\t\tmedian\t\t\t1st Qtl\t\t\t4th Qtl\t\t\tmin\t\t\t\t\t\t\tmax')

        for i in range(1, self._num_features + 1):

            means = []
            mins = []
            maxs = []
            meds = []
            qtl1 = []
            qtl4 = []
            feat = pd.Series()
            for stock in self._stock_data_dict.keys():
                # print('%i NaN Count: %i' % (i, np.isnan(self._stock_data[stock]['I{}'.format(i)]).sum()))
                # print('%i +Inf Count: %i' % (i, np.isinf(self._stock_data[stock]['I{}'.format(i)]).sum()))
                # print('%i -Inf Count: %i' % (i, np.isneginf(self._stock_data[stock]['I{}'.format(i)]).sum()))
                # continue
                means.append(self._stock_data_dict[stock]['I{}'.format(i)].mean())
                mins.append(self._stock_data_dict[stock]['I{}'.format(i)].min())
                maxs.append(self._stock_data_dict[stock]['I{}'.format(i)].max())
                meds.append(self._stock_data_dict[stock]['I{}'.format(i)].median())
                qtl1.append(self._stock_data_dict[stock]['I{}'.format(i)].quantile(0.25))
                qtl4.append(self._stock_data_dict[stock]['I{}'.format(i)].quantile(0.75))

                feat = pd.concat([feat, self._stock_data_dict[stock]['I{}'.format(i)]])
                # ax.scatter(np.ones(len(self._stock_data[stock]['I{}'.format(i)])) * i, self._stock_data[stock]['I{}'.format(i)], s=10, alpha=0.01)

            feat_max = feat.max()
            feat_min = feat.min()
            feat_mean = feat.mean()
            feat_median = feat.median()
            feat_05_quantile = feat.quantile(0.05)
            feat_1st_quantile = feat.quantile(0.25)
            feat_4th_quantile = feat.quantile(0.75)
            feat_95_quantile = feat.quantile(0.95)
            ax.plot((i, i), (feat_05_quantile, feat_95_quantile), 'b-', linewidth=5.0, alpha=0.3)
            ax.plot((i, i), (feat_1st_quantile, feat_4th_quantile), 'b-', linewidth=3.0)
            ax.plot((i, i), (feat_min, feat_1st_quantile), 'b:')
            ax.plot((i, i), (feat_max, feat_4th_quantile), 'b:')
            ax.plot((i - 0.2, i + 0.2), (feat_max, feat_max), 'r-')
            ax.plot((i - 0.2, i + 0.2), (feat_min, feat_min), 'r-')
            ax.plot((i - 0.2, i + 0.2), (feat_mean, feat_mean), 'r-')
            ax.plot((i - 0.2, i + 0.2), (feat_median, feat_median), 'b-')
            ax.plot((i - 0.2, i + 0.2), (feat_1st_quantile, feat_1st_quantile), 'g-')
            ax.plot((i - 0.2, i + 0.2), (feat_4th_quantile, feat_4th_quantile), 'g-')
            means = np.array(means)
            mins = np.array(mins)
            maxs = np.array(maxs)
            meds = np.array(meds)
            qtl1 = np.array(qtl1)
            qtl4 = np.array(qtl4)
            print('%i\t\t'
                  '%+.2f ±%.2f\t\t'
                  '%+.2f ±%.2f\t\t'
                  '%+.2f ±%.2f\t\t'
                  '%+.2f ±%.2f\t\t'
                  '%+.2f (Ø %+.2f ±%.2f)\t\t'
                  '%+.2f (Ø %+.2f ±%.2f)' %
                  (i,
                   feat_mean, means.std(),
                   feat_median, meds.std(),
                   feat_1st_quantile, qtl1.std(),
                   feat_4th_quantile, qtl4.std(),
                   feat_min, mins.mean(), mins.std(),
                   feat_max, maxs.mean(), maxs.std()))
        ax.axhline( 1.0, 0, 1, linestyle='--')
        ax.axhline(-1.0, 0, 1, linestyle='--')
        ax.grid(color='grey', linestyle='--', linewidth=1, alpha=0.3)
        plt.xlim(0, self._num_features + 1)
        plt.ylim(-2, 3)
        plt.xticks(np.arange(1, self._num_features + 1, 1.0))
        plt.show()
        # quit()

    def plot_stock_data(self, stocks: list, cols: list, start_date: str = None, end_date: str = None, blocking=True) -> None:
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        days = mdates.DayLocator()  # every day
        yearsFmt = mdates.DateFormatter('%Y')
        monthsFmt = mdates.DateFormatter('%m')

        fig, ax = plt.subplots()
        for stock in stocks:
            if 'Split' in self._stock_data_dict[stock]:
                for xc in self._stock_data_dict[stock][self._stock_data_dict[stock]['Split'] > 0].index:
                    ax.axvline(x=xc, c='r', linestyle='--', alpha=0.6)
            for col in cols:
                if col in self._stock_data_dict[stock]:
                    if start_date is None:
                        ax.plot(self._stock_data_dict[stock].index.values, self._stock_data_dict[stock][col], label='{} {}'.format(stock, col))
                    elif end_date is None:
                        start_index = self._stock_data_dict[stock].index.searchsorted(pd.Timestamp(start_date))
                        ax.plot(self._stock_data_dict[stock].iloc[start_index:].index.values, self._stock_data_dict[stock].iloc[start_index:][col], label='{} {}'.format(stock, col))
                    else:
                        start_index = self._stock_data_dict[stock].index.searchsorted(pd.Timestamp(start_date))
                        end_index = self._stock_data_dict[stock].index.searchsorted(pd.Timestamp(end_date))
                        ax.plot(self._stock_data_dict[stock].iloc[start_index:end_index].index.values, self._stock_data_dict[stock].iloc[start_index:end_index][col], label='{} {}'.format(stock, col))
                else:
                    print("Column '%s' does not exist." % col)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.legend()

        # format the coords message box
        def price(x):
            return '$%1.2f' % x
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        # ax.format_ydata = price
        ax.grid(True)

        # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
        fig.autofmt_xdate()

        if blocking:
            plt.show()

    def add_predictor(self, predictor_class, params, name: str, force_recalculate=False):
        self._predictors[name] = (predictor_class, params, force_recalculate)

    def tune_predictor(self, predictor: AbstractPredictor):
        print("Tuning Predictor...")
        return predictor.tune(self._stock_data_dict, self._symbols, self._num_features, 'f1', self._trading_frequency, self._training_years, self._trading_days_per_year)

    def run_backtest(self, cash_balance=100000):
        stock_data_by_date = self._stock_data_df.swaplevel().sort_index()
        for (name, (predictor, params, force_recalculate)) in self._predictors.items():
            strategy = SingleValueStrategy(predictor, params, name, cash_balance, self.BUY_SIGNAL, self.SELL_SIGNAL,
                                           self.KEEP_SIGNAL, self._trans_cost_fixed, self._trans_cost_percentual,
                                           self._min_trans, self._backtest_data_path, force_recalculate)
            print('Backtesting %s' % name)
            self._backtests[name] = strategy

            timeline = self._stock_data_df.index.levels[1]
            start_date = timeline[0]

            # Need to track day-index as well because pandas slicing always includes the end!
            first_trading_day = self._training_years * self._trading_days_per_year + self._trading_frequency
            last_trading_day = first_trading_day - self._trading_frequency

            step_count = len(timeline) - first_trading_day
            step_width = 4
            i = 0
            j = step_width
            sys.stdout.write('  Progress: [' + '-' * 40 + ']')
            sys.stdout.flush()
            for trading_day in range(first_trading_day, len(timeline), self._trading_frequency):
                # Progress bar
                i += self._trading_frequency
                if i >= (step_count * step_width / 100):
                    i = 0
                    j += step_width
                    sys.stdout.write('\r  Progress: [' + (j * 40 // 100) * '#' + (40 - j * 40 // 100) * '-' + ']')
                    sys.stdout.flush()

                last_training_date = timeline[last_trading_day]
                trading_date = timeline[trading_day]

                training_data = stock_data_by_date.loc[start_date:last_training_date].swaplevel().sort_index()

                feature_labels = ['I{}'.format(x + 1) for x in range(self._num_features)]
                X = training_data[feature_labels]
                y = training_data[TARGET_COL]

                # Product of splits since the last trading day per stock
                accumulated_splits = pd.Series.to_dict(
                    stock_data_by_date[SPLIT_COL]
                    .loc[timeline[last_trading_day + 1]:timeline[trading_day]]
                    .groupby('Symbol')
                    .prod()
                )
                todays_closes = pd.Series.to_dict(stock_data_by_date[CLOSE_COL].loc[trading_date])
                todays_features = pd.DataFrame.to_dict(stock_data_by_date[feature_labels].loc[trading_date], orient='index')
                if len(todays_closes) == len(self._symbols):
                    strategy.backtest_step(X, y, todays_closes, str(trading_date), accumulated_splits, todays_features, self._training_years, self._trading_days_per_year)
                else:
                    print('\nExcluding incomplete day %s' % trading_date)

                last_trading_day = trading_day

            sys.stdout.write('\n')
            sys.stdout.flush()

            strategy.persist()

    def plot_single_value_strategy_balance(self, name):
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        days = mdates.DayLocator()  # every day
        yearsFmt = mdates.DateFormatter('%Y')
        monthsFmt = mdates.DateFormatter('%m')
        fig, ax = plt.subplots()

        strategy = self._backtests[name]
        for stock in self._symbols:
            balances = pd.Series(strategy._backtests[stock].get_balances_series(), name=strategy.name)

            # if 'Split' in self._stock_data_dict[stock]:
            #     for xc in self._stock_data_dict[stock][self._stock_data_dict[stock]['Split'] > 0].index:
            #         ax.axvline(x=xc, c='r', linestyle='--', alpha=0.6)
            start_index = balances.index.searchsorted(pd.Timestamp(self._start_date))
            end_index = balances.index.searchsorted(pd.Timestamp(self._end_date))
            ax.plot(balances.iloc[start_index:end_index].index.values,
                        balances.iloc[start_index:end_index],
                        label='{}'.format(stock))

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.legend()

        # format the coords message box
        def price(x):
            return '$%1.2f' % x

        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        # ax.format_ydata = price
        ax.grid(True)

        # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
        fig.autofmt_xdate()

        plt.show()

    def plot_single_value_strategy_balances(self):
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        ind = np.arange(len(self._symbols))
        width = 0.8 / len(self._backtests)
        fig, ax = plt.subplots()
        ax.set_ylabel('%')
        ax.set_title('Strategy performance')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(self._symbols)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        # ax.set_ylim(0, 20)
        ax.yaxis.grid()
        rects = []
        for backtest, i in zip([backtest for backtest in self._backtests.values()], range(0, len(self._backtests))):
            rects.append(ax.bar(ind + i * width, backtest.get_performance(self._symbols).values(), width=width, alpha=0.7))  #, color='b')
        ax.legend((rect[0] for rect in rects), (name for name in self._backtests.keys()))

        # autolabel(rects1)
        # autolabel(rects2)

        plt.show()

    def evaluate(self, baseline_predictor_name) -> tuple:
        """
        How often did each registered predictor have the highest last balance?
        How often was each registered predictor better than the baseline predictor?
        :param baseline_predictor_name: Name of the predictor to compare all other predictors to.
        :returns: tuple of dicts (predictor_name->f_scores), (predictor_name->balances)
        """
        results = {name: {'best': 0, 'better_than_baseline': 0} for name in self._backtests.keys()}

        last_balances = {}
        baseline_last_balances = (self._backtests[baseline_predictor_name].get_performance(self._symbols)
                                  if baseline_predictor_name in self._backtests.keys()
                                  else {stock: 0.0 for stock in self._symbols})
        balances_ordered = {}
        for backtest in self._backtests.values():
            last_balances[backtest.name] = backtest.get_performance(self._symbols)
            balances_ordered[backtest.name] = []

        first_trading_day = self._training_years * self._trading_days_per_year + self._trading_frequency
        timeline = self._stock_data_df.index.levels[1]
        stock_data_by_date = self._stock_data_df.swaplevel().sort_index()
        all_trading_targets = stock_data_by_date[TARGET_COL].loc[timeline[range(first_trading_day, len(timeline), self._trading_frequency)]].swaplevel().sort_index()
        num_backtests = len(self._symbols)
        f_scores = {}

        for stock in all_trading_targets.index.levels[0]:
            max_balance_pred_name = ''
            max_balance = 0.0
            for (pred_name, last_bal_per_stock) in last_balances.items():
                last_balance = last_bal_per_stock[stock]
                balances_ordered[pred_name].append(last_balance)
                if last_balance > baseline_last_balances[stock]:
                    results[pred_name]['better_than_baseline'] += 1
                if last_balance > max_balance:
                    max_balance = last_balance
                    max_balance_pred_name = pred_name
            results[max_balance_pred_name]['best'] += 1

        for backtest in self._backtests.values():
            print("\nPredictor {}".format(backtest.name))
            f_scores[backtest.name] = backtest.evaluate(all_trading_targets)
            print("  Fraction of best balances: {:.2f}".format(results[backtest.name]['best'] / num_backtests))
            if baseline_predictor_name in self._backtests.keys():
                print("  Fraction of balances better than baseline: {:.2f}".format(results[backtest.name]['better_than_baseline'] / num_backtests))
        return f_scores, balances_ordered
