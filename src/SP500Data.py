import os
import sys
import pickle
import json
import datetime

from collections import OrderedDict

import pandas as pd
import numpy as np
from pandas import DataFrame

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web


class SP500Data:
    def __init__(self, start_date, end_date, tune_or_simulate, key='pfycsevij29AJXH6Gw_e'):
        os.environ['QUANDL_API_KEY'] = key
        self._dir = os.path.dirname(os.path.realpath(__file__))
        self._data_path = os.path.join(self._dir, 'data/sp500/data')
        self._meta_db_path = os.path.join(self._dir, 'sp500_meta_db.pkl')

        if os.path.isfile(self._meta_db_path):
            with open(self._meta_db_path, 'rb') as file:
                self._meta_db = pickle.load(file)
        else:
            self._meta_db = {}

        self._stocks = self._get_initial_stock_list(tune_or_simulate)

        print("Building Stock DataFrame...")
        prog_len = len(self._stocks)
        step_width = 4
        i = 0
        j = step_width

        sys.stdout.write('  Progress: [' + '-' * 40 + ']')
        sys.stdout.flush()

        df = pd.DataFrame()
        for stock in self._stocks:
            # Progress bar
            i += 1
            if i > (prog_len * step_width / 100):
                i = 0
                j += step_width
                sys.stdout.write('\r  Progress: [' + (j * 40 // 100) * '#' + (40 - j * 40 // 100) * '-' + ']')
                sys.stdout.flush()

            # Is the given date period beyond what is already known to us? Then download new data!
            if self._expand_stock_period(stock, start_date, end_date):
                # In case of download error: delete stock meta entry
                if not self._download_stock(stock):
                    del self._meta_db[stock]
                self._persist_meta_db()

            # Read CSV, build DateTimeIndex and append to DataFrame
            df_tmp = pd.read_csv(self._get_stock_filename(stock), delimiter=',', index_col=False)
            df_tmp['Date'] = pd.to_datetime(df_tmp['Date'], format='%Y-%m-%d')
            df_tmp = df_tmp.sort_values(['Name', 'Date'])
            df = pd.concat([df, df_tmp], axis=0)

        df.set_index(['Date', 'Name'], inplace=True)
        df.sort_index(inplace=True)

        # Filter for given time period
        df = df.loc[str(start_date):str(end_date)]

        date_index = df.index.get_level_values(0).drop_duplicates()

        # Remove stocks that are missing for at least two days in a row or at first or last day
        missing_stocks = {}
        stocks_to_drop = set()
        requested_stocks = set(self._stocks)
        date_count = len(date_index.values)
        missing_values = {}
        for i, d in zip(range(date_count), date_index.values):
            present_stocks = set(df.loc[d].index.values)
            missing_stocks[i] = (requested_stocks - present_stocks) - stocks_to_drop
            for stock in missing_stocks[i]:
                if i == 0 or i == (date_count - 1) or stock in missing_stocks[i - 1]:
                    stocks_to_drop.add(stock)
                    if stock in missing_values.keys():
                        missing_values.pop(stock)
                else:
                    if stock not in missing_values.keys():
                        missing_values[stock] = []
                    missing_values[stock].append(i)
        if len(stocks_to_drop) > 0:
            print("\nDropping stocks: %s" % str(stocks_to_drop))
        df = df.drop(stocks_to_drop, level=1)
        self._stocks = sorted(set(self._stocks) - set(stocks_to_drop))

        # Fill missing values: Repeat last row
        for stock, missing_indices in missing_values.items():
            for missing_index in missing_indices:
                df.loc[(df.index.levels[0][missing_index], stock), :] = df.loc[(df.index.levels[0][missing_index - 1], stock), :]

        df = df.swaplevel().sort_index()

        self._dfs = {}
        for stock in df.index.get_level_values(0).drop_duplicates():
            self._dfs[stock] = df.loc[stock]

    def _get_stock_filename(self, stock):
        return os.path.join(self._data_path, stock + '_data.csv')

    def _download_stock(self, stock):
        """
        Try to Query quandl for a stock, return False on error.
        Dates for the request are taken from _meta_db.
        If there is not entry for the requested stock, return False.
        :param stock: String name of the stock to download.
        :return: True on success, False otherwise.
        """
        if stock not in self._meta_db.keys():
            return False

        (start_date, end_date) = self._meta_db[stock]
        try:
            print("Downloading {} stock data from {} to {}".format(stock, start_date, end_date))
            stock_df = web.QuandlReader('WIKI/' + stock, start_date, end_date).read()
            stock_df['Name'] = stock
            output_name = self._get_stock_filename(stock)
            stock_df.to_csv(output_name)
            return True
        except Exception as e:
            print(e)
            return False

    def _expand_stock_period(self, stock, start_date, end_date):
        """
        Expand the period for the given stock in the _meta_db.
        Afterwards the start_date in db is given start_date or earlier and end_date is given end_date or later.
        :param stock: Name of the stock
        :param start_date: Datetime
        :param end_date: Datetime
        :return: True, if the db has been modified, False otherwise.
        """
        modified = True
        if stock in self._meta_db.keys():
            (db_start_date, db_end_date) = self._meta_db[stock]
            if start_date > db_start_date:
                start_date = db_start_date
            if end_date < db_end_date:
                end_date = db_end_date
            if end_date == db_end_date and start_date == db_start_date:
                modified = False

        self._meta_db[stock] = (start_date, end_date)

        return modified

    def _persist_meta_db(self):
        with open(self._meta_db_path, 'wb') as file:
            pickle.dump(self._meta_db, file)

    def get_components(self):
        return self._stocks

    def get(self):
        """
        Get data for given stock names. If the time period has not been used before, download required stock data.
        :return: Mapping stock name symbol to DataFrame with index on date.
        """

        return self._dfs

    def _get_initial_stock_list(self, tune_or_simulate):
        if not tune_or_simulate:
            # Tuning
            return [
                'ABMD', 'ABT', 'ACN', 'ADBE',
                'C', 'COF', 'COL', 'COO',
                'FL', 'GPC',
                'PKI', 'PLD', 'PNC',
                'TTWO', 'TXN', 'TXT',
                'WBA', 'WDC', 'WEC', 'WFC',
            ]
        else:
            # Simulation
            return [
                'MDR',
                #'RRC',
                'RIG', 'A',
                # 'AAL',
                'AAP', 'AAPL',
                # 'ABBV',
                'ABC',
                # 'ABMD', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK',
                'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV',
                # 'AIZ',
                'AJG', 'AKAM', 'ALB',
                # 'ALGN', 'ALK', 'ALL',
                # 'ALLE', 'ALXN', 'AMAT', 'AMD', 'AME', 'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA',
                # 'APC', 'APD', 'APH', 'ARE', 'ARNC', 'ATVI', 'AVB', 'AVGO',
                'AVY',
                # 'AWK',
                'AXP', 'AZO', 'BA', 'BAC', 'BAX',
                'BBT', 'BBY', 'BDX', 'BEN', 'BF_B', 'BIIB', 'BK', 'BLK', 'BLL', 'BMY',
                # 'BR',
                'BRK_B', 'BSX', 'BWA', 'BXP',
                # 'C', 'CA', 'CAG', 'CAH', 'CAT', 'CB', 'CBG', 'CBOE', 'CBS', 'CCI', 'CCL', 'CDNS', 'CELG', 'CERN', 'CF', 'CFG',
                # 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
                #  'COF', 'COG', 'COL', 'COO', 'COP', 'COST', 'COTY', 'CPB', 'CPRT', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTL', 'CTSH',
                # 'CTXS', 'CVS', 'CVX', 'CXO', 'D', 'DAL', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK',
                'DISH', 'DLTR', 'DOV', 'DRE',
                # 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'EA', 'EBAY', 'ECL', 'ED',
                # 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES', 'ESRX', 'ESS', 'ETFC', 'ETN', 'ETR',
                # 'EVHC',
                'EW', 'EXC', 'EXPD',
                # 'EXPE',
                # 'EXR',
                'F', 'FAST',
                # 'FB', 'FBHS',
                'FCX', 'FDX', 'FE', 'FFIV', 'FIS',
                # 'FISV', 'FITB', 'FL', 'FLIR', 'FLR', 'FLS', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRT', 'FTI', 'FTV', 'GD', 'GE',
                # 'GGP', 'GILD', 'GIS', 'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GPS', 'GRMN', 'GS', 'GT', 'GWW', 'HAL',
                # 'HAS', 'HBAN', 'HBI', 'HCA', 'HCN', 'HCP', 'HD', 'HES', 'HFC', 'HIG', 'HII', 'HOG', 'HOLX', 'HON', 'HP', 'HPE',
                # 'HPQ', 'HRB', 'HRL', 'HRS', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IDXX', 'IFF', 'ILMN', 'INCY', 'INTC',
                'INTU', 'IP', 'IPG',
                # 'IPGP',
                'IR', 'IRM',
                # 'ISRG',
                'IT', 'ITW', 'IVZ', 'JBHT', 'JCI', 'JEC', 'JNJ', 'JNPR',
                # 'JPM', 'JWN', 'K', 'KEY', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KORS', 'KR', 'KSS', 'KSU', 'L',
                'LB', 'LEG', 'LEN', 'LH',
                #'LKQ',
                'LLL', 'LLY',
                #'LMT',
                'LNC', 'LNT', 'LOW', 'LRCX', 'LUK', 'LUV',
                #'LYB',
                'M',
                # 'MA',
                'MAA', 'MAC', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK',
                # 'MKC', 'MLM', 'MMC', 'MMM', 'MO', 'MOS', 'MPC', 'MRK', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB',
                # 'MTD', 'MU', 'MYL', 'NBL', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 'NKTR', 'NLSN', 'NOC',
                # 'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWS', 'NWSA', 'O', 'OKE', 'OMC', 'ORCL', 'ORLY',
                # 'OXY', 'PAYX', 'PBCT', 'PCAR', 'PCG', 'PCLN', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG',
                # 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PX', 'PXD',
                # 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RHT', 'RJF', 'RL', 'RMD', 'ROK', 'ROP',
                # 'ROST', 'RSG', 'RTN', 'SBAC', 'SBUX', 'SCG', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SLG', 'SNA', 'SNPS',
                # 'SO', 'SPG', 'SPGI', 'SRCL', 'SRE', 'STI', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYMC', 'SYY',
                'T',
                #'TAP', 'TDG', 'TEL', 'TGT', 'TIF', 'TJX', 'TMK', 'TMO', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS',
                # 'TTWO', 'TWTR', 'TXN', 'TXT', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM', 'UNP', 'UPS', 'URI',
                # 'USB', 'UTX', 'V', 'VAR', 'VFC', 'VIAB', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAT',
                # 'WBA', 'WDC', 'WEC', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT', 'WR', 'WRK', 'WU', 'WY', 'WYNN', 'XEC', 'XEL',
                'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'YUM', 'ZBH', 'ZION']

    def print_stats(self):
        data = self.get()
        base_day = 0
        stock_data_df = pd.concat(list(data.values()), keys=list(data.keys()), names=['Symbol'])
        delta_adjclose = stock_data_df['AdjClose'].groupby('Symbol').apply(lambda df: df.iloc[-1] / df.iloc[base_day])
        delta_adjclose = delta_adjclose.sort_values()


        print("\nAll stocks sorted by relative performance:")
        print(delta_adjclose)
        print(len(delta_adjclose))

        lower_than_inflation = delta_adjclose[delta_adjclose < 1.354]
        print("\nNumber of stocks with performance of less than inflation of (101.96%)^15: {}".format(
            len(lower_than_inflation)))
        print(lower_than_inflation)

        negative_performance = delta_adjclose[delta_adjclose < 1.0]
        print("\nNumber of stocks with performance of less than 100%: {}".format(len(negative_performance)))
        print(negative_performance)

        doubled = delta_adjclose[delta_adjclose >= 2.0]
        print("\nNumber of stocks with performance of at least 200% (5% per year): {}".format(len(doubled)))
        print(doubled)

        print("\nNumber of stocks: %i" % len(delta_adjclose))
        print("Mean: {:.0%}\nMedian: {:.0%}\n25-Qtl: {:.0%}\n75-Qtl: {:.0%}\nMin: {:.0%}\nMax: {:.0%}"
              .format(delta_adjclose.mean(), delta_adjclose.median(), delta_adjclose.quantile(0.25),
                      delta_adjclose.quantile(0.75), delta_adjclose.min(), delta_adjclose.max()))


if __name__ == '__main__':

    end_date = datetime.date(2017, 5, 31)
    start_date = datetime.date(2002, 6, 3)

    sp500 = SP500Data(start_date, end_date, tune_or_simulate=1)
    # stocks = sp500.get_components()
    # data = sp500.get()
    # symbols_to_exclude = []
    # for stock in stocks:
    #     index_min = data[stock].index.min()
    #     index_max = data[stock].index.max()
    #     if index_min.date() != start_date or index_max.date() != end_date:
    #         print('Excluding %s' % stock)
    #         symbols_to_exclude.append(stock)
    #         data.pop(stock)
    #         continue

    sp500.print_stats()
