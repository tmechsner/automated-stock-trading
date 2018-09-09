import os
import sys

import technical_indicators
import bovespa_data

from SAMkNN.SAMKNN.SAMKNNClassifier import SAMKNN

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


vol_col = '#trades'  # try also '#titles traded' and 'volume'


def annotate_splits(dfs: dict, threshold: float) -> dict:
    """
    Finds sudden price drops and rises (close to next day open) of more than the given threshold.
    :param dfs:
    :param threshold:
    :return: {stock_name:string -> stock_data:DataFrame} with Split column that is True if in the last night there was
    a drop greater than threshold, False otherwise.
    """
    result = {}

    for symbol, stock_df in dfs.items():
        splits = pd.Series(stock_df['open'] / stock_df['close'].shift(1), name='Split')
        splits[(splits.abs() > threshold) & (splits.abs() < (1 / threshold)) | (stock_df['open'] < 2)] = 0.0
        adj_close = pd.Series(stock_df['close'].copy(), name='adj_close')
        adj_low = pd.Series(stock_df['low'].copy(), name='adj_low')
        adj_high = pd.Series(stock_df['high'].copy(), name='adj_high')
        adj_open = pd.Series(stock_df['open'].copy(), name='adj_open')
        cumulative_split = 1.0
        for date, _ in adj_close.items():
            if splits[date] > 0.0:
                cumulative_split *= splits[date]
            adj_close[date] /= cumulative_split
            adj_low[date] /= cumulative_split
            adj_high[date] /= cumulative_split
            adj_open[date] /= cumulative_split
        result[symbol] = pd.concat([stock_df, splits, adj_close, adj_low, adj_high, adj_open], axis=1)
        print('\n{} number of >{:.0%} deltas: {}'.format(symbol, threshold, splits[splits > 0].size))
        # print(splits[splits > 0].index)
        print(splits[splits > 0])

    print('\n')
    return result


def build_tech_ind(dfs: dict) -> dict:
    """
    Build technical indicators and append them to the DataFrames as columns.
    :param dfs: Dict containing DataFrames by symbol containing all stock data as provided by group_by_stockname()
    :return: dict: {stock_name:string -> stock_data:DataFrame} with additional feature columns
    """
    result = {}
    for symbol, stock_df in dfs.items():
        close_col = 'adj_close' if 'adj_close' in stock_df else 'close'
        open_col  = 'adj_open'  if 'adj_open'  in stock_df else 'open'
        low_col   = 'adj_low'   if 'adj_low'   in stock_df else 'low'
        high_col  = 'adj_high'  if 'adj_high'  in stock_df else 'high'
        pmas = technical_indicators.sma(stock_df, 10, 'PMA_S', col=close_col)
        pmal = technical_indicators.sma(stock_df, 21, 'PMA_L', col=close_col)
        vmas = technical_indicators.sma(stock_df, 10, 'VMA_S', col=vol_col)
        vmal = technical_indicators.sma(stock_df, 21, 'VMA_L', col=vol_col)
        rsi = technical_indicators.rsi(stock_df, 14, col=close_col)
        sto = technical_indicators.sto(stock_df, 14, 3, col=close_col, col_low=low_col, col_high=high_col)
        bb = technical_indicators.bbands(stock_df, 8, 2, col=close_col)
        result[symbol] = pd.concat([stock_df, pmas, pmal, vmas, vmal, rsi, sto, bb], axis=1)
    return result


def build_features(dfs: dict, two_features_only: bool = False) -> dict:
    """
    Build features and append them to the DataFrames as columns.
    :param dfs: Dict containing DataFrames by symbol containing all stock data as provided by group_by_stockname()
    :return: dict: {stock_name:string -> stock_data:DataFrame} with additional feature columns
    """
    result = {}
    for symbol, stock_df in dfs.items():
        close_col = 'adj_close' if 'adj_close' in stock_df else 'close'
        open_col  = 'adj_open'  if 'adj_open'  in stock_df else 'open'
        low_col   = 'adj_low'   if 'adj_low'   in stock_df else 'low'
        high_col  = 'adj_high'  if 'adj_high'  in stock_df else 'high'
        target = pd.Series(stock_df[close_col].diff(-1), name='target')
        target[target < 0] = -1  # Next day's close is greater -> Buy
        target[target > 0] = 1   # Next day's close is smaller -> Sell
        if two_features_only:
            target2 = pd.Series(stock_df[close_col].diff(-2), name='target2')
            target2[target2 < 0] = -1  # Second next day's close is greater -> Buy
            target2[target2 >= 0] = 1   # Second next day's close is smaller / bigger -> Sell
            target[target == 0] = target2[target == 0]  # Same price next day? Look at day after tomorrow
        i1  = pd.Series(stock_df['RSI'], name='I1')
        i2  = pd.Series((stock_df[close_col] - stock_df['BB_U']) / stock_df['BB_U'], name='I2')
        i3  = pd.Series((stock_df[close_col] - stock_df['BB_L']) / stock_df['BB_L'], name='I3')
        i4  = pd.Series( stock_df['%K'], name='I4')
        i5  = pd.Series( stock_df['%D'], name='I5')
        i6  = pd.Series( stock_df['%K'].diff(), name='I6')
        i7  = pd.Series( stock_df['%D'].diff(), name='I7')
        i8  = pd.Series( stock_df[close_col].diff() / stock_df[close_col].shift(1), name='I8')
        i9  = pd.Series((stock_df[close_col] - stock_df[low_col]) / (stock_df[high_col] - stock_df[low_col]), name='I9')
        i10 = pd.Series((stock_df['PMA_S'] - stock_df['PMA_S'].shift(1)) / stock_df['PMA_S'].shift(1), name='I10')
        i11 = pd.Series((stock_df['PMA_L'] - stock_df['PMA_L'].shift(1)) / stock_df['PMA_L'].shift(1), name='I11')
        i12 = pd.Series((stock_df['PMA_S'] - stock_df['PMA_L'].shift(1)) / stock_df['PMA_L'].shift(1), name='I12')
        i13 = pd.Series((stock_df[close_col] - stock_df['PMA_L']) / stock_df['PMA_L'], name='I13')
        i14 = pd.Series((stock_df[close_col] - stock_df[close_col].rolling(6).min()) / stock_df[close_col].rolling(6).min(), name='I14')
        i15 = pd.Series((stock_df[close_col] - stock_df[close_col].rolling(6).max()) / stock_df[close_col].rolling(6).max(), name='I15')
        i16 = pd.Series((stock_df[vol_col] - stock_df[vol_col].shift(1)) / stock_df[vol_col].shift(1), name='I16')
        i17 = pd.Series((stock_df['VMA_S'] - stock_df['VMA_S'].shift(1)) / stock_df['VMA_S'].shift(1), name='I17')
        i18 = pd.Series((stock_df['VMA_L'] - stock_df['VMA_L'].shift(1)) / stock_df['VMA_L'].shift(1), name='I18')
        i19 = pd.Series((stock_df['VMA_S'] - stock_df['VMA_L'].shift(1)) / stock_df['VMA_L'].shift(1), name='I19')
        i20 = pd.Series((stock_df[vol_col] - stock_df['VMA_L']) / stock_df['VMA_L'], name='I20')
        i21 = pd.Series((stock_df[vol_col] - stock_df[vol_col].rolling(6).min()) / stock_df[vol_col].rolling(6).min(), name='I21')
        i22 = pd.Series((stock_df[vol_col] - stock_df[vol_col].rolling(6).max()) / stock_df[vol_col].rolling(6).max(), name='I22')
        result[symbol] = pd.concat([stock_df, target, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22], axis=1)
    return result


def handle_outliers(dfs: dict) -> dict:
    """
    Drop rows containing NaN. This value is likely to appear in early rows as some features require several days of
    historical data to be present for being calculated.
    Replace inf and -inf with 0.0.
    :param dfs:
    :return:
    """
    result = dfs.copy()
    for symbol, df in dfs.items():
        result[symbol] = result[symbol].dropna()
        result[symbol] = result[symbol].replace([np.inf, -np.inf], 0.0)

    return result


def normalize_data(dfs: dict) -> dict:
    result = dfs.copy()
    for symbol, stock_df in result.items():
        for i in range(1,23):
            feat = result[symbol]['I{}'.format(i)]
            result[symbol]['I{}'.format(i)] = (feat - feat.mean()) / feat.std()
    return result


def date_range(dfs: dict, start: str, stop: str = None) -> dict:
    result = {}
    for symbol, df in dfs.items():
        start_index = dfs[symbol].index.searchsorted(pd.Timestamp(start))
        if stop is None:
            result[symbol] = dfs[symbol].iloc[start_index:]
        else:
            stop_index = dfs[symbol].index.searchsorted(pd.Timestamp(stop))
            result[symbol] = dfs[symbol].iloc[start_index:stop_index]

    return result


def print_feature_stats(dfs: dict):
    print('ft\t\tmean mean\tmean std\tmin mean\tmin std\t\tmax mean\tmax std')
    for x in range(1, 23):
        means = []
        mins = []
        maxs = []
        for stock in dfs.keys():
            means.append(dfs[stock]['I{}'.format(x)].mean())
            mins.append(dfs[stock]['I{}'.format(x)].min())
            maxs.append(dfs[stock]['I{}'.format(x)].max())
        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)
        print(x, '%.3f' % means.mean(), '%.3f' % means.std(), '%.3f' % mins.mean(), '%.3f' % mins.std(),
              '%.3f' % maxs.mean(), '%.3f' % maxs.std(), sep='\t\t')


def plot_stock_data(data: dict, stocks: list, cols: list, start_date: str = None, end_date: str = None, blocking = True) -> None:
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    days = mdates.DayLocator()  # every day
    yearsFmt = mdates.DateFormatter('%Y')
    monthsFmt = mdates.DateFormatter('%m')


    fig, ax = plt.subplots()
    for stock in stocks:
        if 'Split' in data[stock]:
            for xc in data[stock][data[stock]['Split'] > 0].index:
                ax.axvline(x=xc, c='r', linestyle='--', alpha=0.6)
        for col in cols:
            if col in data[stock]:
                if start_date is None:
                    ax.plot(data[stock].index.values, data[stock][col], label='{} {}'.format(stock, col))
                elif end_date is None:
                    start_index = data[stock].index.searchsorted(pd.Timestamp(start_date))
                    ax.plot(data[stock].iloc[start_index:].index.values, data[stock].iloc[start_index:][col], label='{} {}'.format(stock, col))
                else:
                    start_index = data[stock].index.searchsorted(pd.Timestamp(start_date))
                    end_index = data[stock].index.searchsorted(pd.Timestamp(end_date))
                    ax.plot(data[stock].iloc[start_index:end_index].index.values, data[stock].iloc[start_index:end_index][col], label='{} {}'.format(stock, col))
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


def output_metrics(X_test, buy_sig_pred, buy_sig_test, dummy_cls, knn, sell_sig_pred, sell_sig_test,
                   y_test, print_class_report=True, plot_roc=False):

    # print('Train/Pred sell signals:', len(y_test[sell_sig_test]), len(preds[sell_sig_pred]))
    # print('Train/Pred buy signals:', len(y_test[buy_sig_test]), len(preds[buy_sig_pred]))
    print('\nkNN Buy Accuracy: %.2f' % ((buy_sig_test == buy_sig_pred).sum() / buy_sig_test.size))
    print('kNN Sell Accuracy: %.2f' % ((sell_sig_test == sell_sig_pred).sum() / sell_sig_test.size))
    print('Majority Accuracy: %.2f' % dummy_cls.score(X_test, y_test))

    if print_class_report:
        print(metrics.classification_report(buy_sig_test, buy_sig_pred, target_names=['S/H', 'Buy']))
        print(metrics.classification_report(sell_sig_test, sell_sig_pred, target_names=['B/H', 'Sell']))

    if plot_roc:
        knn_buy_fpr, knn_buy_tpr, _ = metrics.roc_curve(buy_sig_test, (knn.predict_proba(X_test)[:, 1]))
        knn_buy_auc = metrics.auc(knn_buy_fpr, knn_buy_tpr)
        knn_sell_fpr, knn_sell_tpr, _ = metrics.roc_curve(sell_sig_test, (knn.predict_proba(X_test)[:, 1]))
        knn_sell_auc = metrics.auc(knn_sell_fpr, knn_sell_tpr)
        plt.figure()
        line_width = 2
        plt.plot(knn_buy_fpr, knn_buy_tpr, color='darkorange',
                 lw=line_width, label='kNN buy signal ROC curve (area = %0.4f)' % knn_buy_auc)
        plt.plot(knn_sell_fpr, knn_sell_tpr, color='blue',
                 lw=line_width, label='kNN sell signal ROC curve (area = %0.4f)' % knn_sell_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()


def doBacktest(stock_data, training_years, trading_days_per_year, preds, trans_cost, min_trans):

    cash_balance = 100000
    num_shares = 0
    balances = []
    num_trans = 0
    pred_len = len(list(preds.iloc[trading_days_per_year * training_years:].items()))
    step_width = 4
    i = 0
    j = step_width

    sys.stdout.write('  Progress: [' + '-' * 40 + ']')
    sys.stdout.flush()
    for date, sig in preds.iloc[trading_days_per_year * training_years:].items():
        i += 1
        if i > (pred_len * step_width / 100):
            i = 0
            j += step_width
            sys.stdout.write('\r  Progress: [' + (j * 40 // 100) * '#' + (40 - j * 40 //100) * '-' + ']')
            sys.stdout.flush()

        close_ = stock_data.loc[date]['close']

        # Adjust shares count according to split
        if 'Split' in stock_data.loc[date]:
            split_ = stock_data.loc[date]['Split']
            if split_ > 0:
                num_shares = num_shares // split_

        if sig == -1:
            # Buy
            shares_to_buy = (cash_balance - trans_cost) // close_
            if cash_balance > 0 and shares_to_buy >= min_trans:
                num_shares += shares_to_buy
                cash_balance -= shares_to_buy * close_
                cash_balance -= trans_cost
                num_trans += 1
        elif sig == 1:
            # Sell
            if num_shares > 0:
                cash_balance += close_ * num_shares
                cash_balance -= trans_cost
                num_shares = 0
                num_trans += 1

        balance = cash_balance + num_shares * close_
        balances.append(balance)

    sys.stdout.write('\n')
    sys.stdout.flush()

    # buy_sig_pred = (preds == -1)
    # sell_sig_pred = (preds == 1)
    #
    # buy_sig_test = (stock_data['target'] == -1)
    # sell_sig_test = (stock_data['target'] == 1)
    # output_metrics(X_test, buy_sig_pred, buy_sig_test, dummy_cls, knn, sell_sig_pred, sell_sig_test,
    #                y_test, print_class_report=False, plot_roc=False)
    accuracy = metrics.accuracy_score(stock_data['target'].iloc[trading_days_per_year * training_years:], preds.iloc[trading_days_per_year * training_years:])
    print('   Accuracy: %.2f' % accuracy)

    balances_series = pd.Series(np.array(balances), name='Balance')
    balances_series.index = stock_data.iloc[trading_days_per_year * training_years:].index
    return balances_series


def backtest_oliveira(stock_data, training_years, test_years, trading_days_per_year, min_trans, trans_cost, k=8):

    # No trading in the first trading_years years -> set predictions to zero
    all_preds = pd.Series(np.zeros(trading_days_per_year * training_years), name='Signal')

    max_trade_years = (len(stock_data) // trading_days_per_year) + 1 - training_years
    days_left = 0
    for current_year in range(training_years, training_years + max_trade_years):
        start_train = trading_days_per_year * (current_year - training_years)
        end_train = trading_days_per_year * current_year
        start_test = end_train
        end_test = trading_days_per_year * (current_year + test_years)

        if end_train >= len(stock_data):
            days_left = end_test - len(stock_data)
            break

        if end_test >= len(stock_data):
            end_test = len(stock_data)

        train_data = stock_data.iloc[start_train:end_train]
        test_data = stock_data.iloc[start_test:end_test]

        X_train = train_data[['I{}'.format(x) for x in range(1, 23)]]
        y_train = train_data['target']

        X_test = test_data[['I{}'.format(x) for x in range(1, 23)]]
        y_test = test_data['target']

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        dummy_cls = DummyClassifier('most_frequent')
        dummy_cls.fit(X_train, y_train)

        preds = pd.Series(knn.predict(X_test), name='Signal')
        all_preds = pd.concat([all_preds, preds])

    if days_left > 0:
        to_fill = range(trading_days_per_year * (training_years + max_trade_years + test_years) - days_left, len(stock_data['target']))
        print(to_fill)
        for _ in to_fill:
            all_preds.append([1])
    all_preds.index = stock_data.index

    return doBacktest(stock_data, training_years, trading_days_per_year, all_preds, trans_cost, min_trans)


def backtest_samknn(stock_data, training_years, trade_during_training, trading_days_per_year, hyperParams, trans_cost, min_trans):

    classifier = SAMKNN(n_neighbors=hyperParams['nNeighbours'],
                        maxSize=hyperParams['maxSize'],
                        knnWeights=hyperParams['knnWeights'],
                        recalculateSTMError=hyperParams['recalculateSTMError'],
                        useLTM=hyperParams['useLTM'])

    X = stock_data[['I{}'.format(x) for x in range(1, 23)]]
    y = stock_data['target']

    predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(np.array(X), np.array(y), np.unique(y))

    preds = pd.Series(predictedLabels, name='Signal')
    if not trade_during_training:
        preds[0:trading_days_per_year * training_years - 1] = 0  # for fairness: Don't trade in knn's training years
    preds.index = X.index

    return doBacktest(stock_data, training_years, trading_days_per_year, preds, trans_cost, min_trans)


def backtest_buy_and_hold(stock_data, training_years, trade_during_training, trading_days_per_year, trans_cost, min_trans):
    tmp = np.zeros(len(stock_data))
    if trade_during_training:
        tmp[0] = -1  # for fairness: Don't trade in knn's training years
    else:
        tmp[trading_days_per_year * training_years] = -1  # for fairness: Don't trade in knn's training years
    preds = pd.Series(tmp, name='Signal')
    preds.index = stock_data.index

    return doBacktest(stock_data, training_years, trading_days_per_year, preds, trans_cost, min_trans)


def stock_pred():
    shares_of_interest = [
        'AMBV4',  # 3 no data
        'ARCZ6',  # 0
        'BBAS3',  # 1c
        'BBDC4',  # 5c
        'CMIG4',  # 3c
        'CRUZ3',  # 1 no data
        'CSNA3',  # 3c
        'ELET6',  # 1c
        'ITAU4',  # 3c (real name: ITUB4)
        'ITSA4',  # 0c
        'NETC4',  # 1 no data
        'PETR4',  # 3c
        'TNLP4',  # 0 no data
        'USIM5',  # 4c
        'VALE5'   # 3c
    ]

    load_prepared_data = True
    recalculate_bh = True
    recalculate_knn = False
    recalculate_sam = False
    trade_during_training_sam = False
    trade_during_training_bh = False
    dataset = 'bovespa'
    k = 5
    two_classes_only = False
    start_date = '1998-04-01'
    end_date = '2009-03-09'

    pwd = os.path.realpath(__file__)
    orig_data_path = os.path.join(os.path.dirname(pwd), 'data/{}/orig_data'.format(dataset))
    transformed_data_path = os.path.join(os.path.dirname(pwd), 'data/{}/data'.format(dataset))
    annotated_data_path = os.path.join(os.path.dirname(pwd), 'data/{}/annotated_data/{}_classes'.format(dataset, 2 if two_classes_only else 3))
    backtest_data_path = os.path.join(os.path.dirname(pwd), 'data/{}/backtests/{}_classes/k_{}'.format(dataset, 2 if two_classes_only else 3, k))


    if load_prepared_data and os.listdir(annotated_data_path) != []:
        stock_data_complete = {}
        for stock in shares_of_interest:
            stock_data_complete[stock] = pd.read_pickle(os.path.join(annotated_data_path, '{}.pkl'.format(stock)))
    else:
        bovespa_data.convert_to_csv(shares_of_interest, orig_data_path, transformed_data_path)
        stock_data_complete = bovespa_data.load_data(transformed_data_path)
        stock_data_complete = annotate_splits(stock_data_complete, 0.69)
        stock_data_complete = build_tech_ind(stock_data_complete)
        stock_data_complete = build_features(stock_data_complete, two_features_only=two_classes_only)
        stock_data_complete = handle_outliers(stock_data_complete)
        # stock_data_complete = normalize_data(stock_data_complete)
        for stock in shares_of_interest:
            stock_data_complete[stock].to_pickle(os.path.join(annotated_data_path, '{}.pkl'.format(stock)))

    print('\n{} to {}'.format(start_date, end_date))
    stock_data = date_range(stock_data_complete, start_date, end_date)

    print_feature_stats(stock_data)
    # index = 6
    # splits_ = stock_data[shares_of_interest[index]]['Split']
    # print(splits_[splits_ > 0])
    # plot_stock_data(stock_data, [shares_of_interest[index]], ['BB_L', 'close', 'adj_close', 'I3'])
    # quit()

    trans_cost = 5  # Fixed transaction cost
    min_trans = 100  # Minimum of shares to be bought
    training_years = 3  # Number of years to use as training basis -> don't trade in the first training_years years
    trading_days_per_year = 246  # Days per year

    bh_final_balances = np.array([])
    knn_final_balances = np.array([])
    sam_final_balances = np.array([])
    bh_best = 0
    knn_best = 0
    sam_best = 0

    recalculate_bh |= os.listdir(backtest_data_path) == []
    recalculate_knn |= os.listdir(backtest_data_path) == []
    recalculate_sam |= os.listdir(backtest_data_path) == []
    for stock in shares_of_interest:
        print('\nStock: %s (%i of %i)' % (stock, shares_of_interest.index(stock) + 1, len(shares_of_interest)))

        if recalculate_knn:
            print('Backtesting kNN...')
            knn_balances = backtest_oliveira(stock_data[stock], training_years, 1, trading_days_per_year, min_trans, trans_cost, k)
            knn_balances.rename('knn_balance')
            knn_balances.to_pickle(os.path.join(backtest_data_path, 'knn_{}.pkl'.format(stock)))
        else:
            knn_balances = pd.read_pickle(os.path.join(backtest_data_path, 'knn_{}.pkl'.format(stock)))
        stock_data[stock] = pd.concat([stock_data[stock], knn_balances], axis=1)
        last_knn_balance = knn_balances.iloc[-1]
        knn_final_balances = np.append(knn_final_balances, [last_knn_balance / 100000])

        if recalculate_sam:
            print('Backtesting SAM...')
            hyperParams = {'maxSize': 5000, 'nNeighbours': k, 'knnWeights': 'distance', 'recalculateSTMError': False,
                           'useLTM': True}
            sam_balances = backtest_samknn(stock_data[stock], training_years, trade_during_training_sam, trading_days_per_year, hyperParams, trans_cost, min_trans)
            sam_balances.rename('sam_balance')
            sam_balances.to_pickle(os.path.join(backtest_data_path, 'sam_{}.pkl'.format(stock)))
        else:
            sam_balances = pd.read_pickle(os.path.join(backtest_data_path, 'sam_{}.pkl'.format(stock)))
        stock_data[stock] = pd.concat([stock_data[stock], sam_balances], axis=1)
        last_sam_balance = sam_balances.iloc[-1]
        sam_final_balances = np.append(sam_final_balances, [last_sam_balance / 100000])

        if recalculate_bh:
            print('Backtesting Buy and Hold...')
            bh_balances = backtest_buy_and_hold(stock_data[stock], training_years, trade_during_training_bh, trading_days_per_year, trans_cost, min_trans)
            bh_balances.rename('bh_balance')
            bh_balances.to_pickle(os.path.join(backtest_data_path, 'buy_and_hold_{}.pkl'.format(stock)))
        else:
            bh_balances = pd.read_pickle(os.path.join(backtest_data_path, 'buy_and_hold_{}.pkl'.format(stock)))
        stock_data[stock] = pd.concat([stock_data[stock], bh_balances], axis=1)
        last_bh_balance = bh_balances.iloc[-1]
        bh_final_balances = np.append(bh_final_balances, [last_bh_balance / 100000])

        max_balance = max(last_bh_balance, last_knn_balance, last_sam_balance)
        # max_balance = max(last_bh_balance, last_sam_balance)
        # max_balance = max(last_bh_balance, last_knn_balance)
        if last_bh_balance == max_balance:
            bh_best += 1
        elif last_knn_balance == max_balance:
            knn_best += 1
        elif last_sam_balance == max_balance:
            sam_best += 1
        # print('--- %s Result ---' % stock)
        # print('Buy&Hold: {:,.2f}'.format(last_bh_balance))
        # print('kNN Pred: {:,.2f}'.format(last_knn_balance))
        # plot_stock_data(stock_data, [stock], ['close', 'adj_close'], start_date = '1998-04-01', end_date = '2009-03-09', blocking=False)
        # plot_stock_data(stock_data, [stock], ['B&H Balance', 'kNN Balance'], start_date = '1998-04-01', end_date = '2009-03-09')
        # quit()

    print('\nB&H vs. kNN vs. SAM: %i - %i - %i\n' % (bh_best, knn_best, sam_best))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    ind = np.arange(len(shares_of_interest))
    width = 0.3
    fig, ax = plt.subplots()
    ax.set_ylabel('%')
    ax.set_title('Strategy performance')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(shares_of_interest)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    # ax.set_ylim(0, 20)
    ax.yaxis.grid()
    rects1 = ax.bar(ind, bh_final_balances, width, color='b')
    rects2 = ax.bar(ind + width, knn_final_balances, width, color='y')
    rects3 = ax.bar(ind + 2 * width, sam_final_balances, width, color='g')
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Buy & Hold', 'k-NN', 'SAM k-NN'))

    # autolabel(rects1)
    # autolabel(rects2)

    plt.show()


if __name__ == '__main__':
    # visual_techind_test()

    stock_pred()
