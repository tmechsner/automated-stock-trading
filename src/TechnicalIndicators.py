import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


COL_CLOSE = 'close'
COL_OPEN = 'open'
COL_HIGH = 'high'
COL_LOW = 'low'


def sma(df: pd.DataFrame, n: int, name: str = 'MA', col: str = COL_CLOSE) -> pd.Series:
    """
    Calculates a simple moving average.
    :param df: TimeSeries to calculate on.
    :param n: Size of the sliding window.
    :param name: Name of the resulting Series.
    :param col: Column to calculate MA on.
    :return: TimeSeries of averages named MA.
    """

    return pd.Series(df[col].rolling(n).mean(), name=name)


def rsi(df: pd.DataFrame, n: int = 14, col: str = COL_CLOSE) -> pd.Series:
    """
    Calculates the Relative Strength Index based on SMA. Range between 0 and 100.
    It is interpreted as overbought for values above 70 and oversold for values below 30.
    :param df: TimeSeries to calculate on.
    :param n: Number of periods to consider (common value is 14).
    :return: TimeSeries of RSI values named RSI
    """

    deltas = pd.Series(df[col]).diff()

    gains, losses = deltas.copy(), deltas.copy()

    gains[gains < 0] = 0
    losses[losses >= 0] = 0

    avg_gain = gains.rolling(n).mean()
    avg_loss = losses.abs().rolling(n).mean()
    avg_loss[avg_loss == 0] = avg_loss.min() * 0.1  # Avoid division by zero

    rs = avg_gain / avg_loss

    return pd.Series(100 - 100 / (rs + 1), name='RSI')


def sto(df: pd.DataFrame, n_k: int = 14, n_d: int = 3, col: str = COL_CLOSE, col_low: str = COL_LOW, col_high: str = COL_HIGH) -> pd.DataFrame:
    """
    Calculates the stochastics oscillators %K and %D without smoothing. Range between 0 and 100.
    Values at the top of the range indicate accumulation (buying pressure),
    those at the bottom indicate distribution (selling pressure).
    :param df: TimeSeries to calculate on.
    :param n_k: Period of consideration for %K (common value is 14).
    :param n_d: Periods to average of %K for %D (common value is 3).
    :return: DataFrame containing two columns of TimeSeries named %K and %D.
    """
    recent_close = df[col]
    lowest_low = df[col_low].rolling(n_k).min()
    highest_high = df[col_high].rolling(n_k).max()
    sto_k = pd.Series(data=100 * (recent_close - lowest_low) / (highest_high - lowest_low), name='%K')
    sto_d = pd.Series(data=sto_k.rolling(n_d).mean(), name='%D')
    return pd.concat([sto_k, sto_d], axis=1)


def bbands(df: pd.DataFrame, n: int = 20, n_std: int = 2, col: str = COL_CLOSE) -> pd.DataFrame:
    """
    Calculates three bollinger bands: SMA(n) in the middle, SMA(n) + n_std * standard deviation as upper band and
    SMA(n) - n_std * standard deviation as lower band.
    :param df: Time Series to calculate on.
    :param n: Period of consideration for SMAs (common value is 20).
    :param n_std: Number of standard deviations to add or substract for upper and lower bands (recommended value is 2).
    :return: DataFrame containing three columns of TimeSeries named BB_U, BB_M and BB_L
    """
    bb_m = sma(df, n, 'MA', col).rename('BB_M_{}_{}'.format(n, n_std))
    std = df[col].rolling(n).std()
    bb_u = pd.Series(bb_m + n_std * std, name='BB_U'.format(n, n_std))
    bb_l = pd.Series(bb_m - n_std * std, name='BB_L'.format(n, n_std))
    return pd.concat([bb_u, bb_m, bb_l], axis=1)


def visual_techind_test():
    n = 100
    quant = 10
    t_raw = np.linspace(0, n * quant, n * quant) * (1 / quant)
    t = np.linspace(0, n, n)
    raw = pd.Series(t_raw * 0.03 + np.cos(t_raw * 0.15) + np.sin((3 * t_raw - n * quant * 0.5) ** 2 * 0.1)) \
        .rolling(3).mean().dropna()
    open = raw.rename(COL_OPEN).iloc[:-quant:quant].reset_index(drop=True)
    close = raw.rename(COL_CLOSE).iloc[quant - 1::quant].reset_index(drop=True)
    high = raw.rename(COL_HIGH).rolling(quant).max().iloc[quant - 1::quant].reset_index(drop=True)
    low = raw.rename(COL_LOW).rolling(quant).min().iloc[quant - 1::quant].reset_index(drop=True)
    df = pd.concat([open, close, high, low], axis=1)

    rs = rsi(df, 14)
    so = sto(df, 14, 3)
    stok = so['%K']
    stod = so['%D']
    bb = bbands(df, 14, 2)
    bb_u = bb[bb.columns.values[0]]
    bb_m = bb[bb.columns.values[1]]
    bb_l = bb[bb.columns.values[2]]

    plt.figure()

    ax = plt.subplot(311)
    ax.plot(df.index.get_values(), df[COL_CLOSE].tolist(), label='time series 1')
    ax.plot(df.index.get_values(), bb_u.tolist(), label='BBand upper')
    ax.plot(df.index.get_values(), bb_m.tolist(), label='BBand middle')
    ax.plot(df.index.get_values(), bb_l.tolist(), label='BBand lower')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()

    ax = plt.subplot(312)
    ax.plot(rs.index.get_values(), rs.tolist(), label=rs.name)
    ax.set_xlabel('t')
    ax.set_ylabel('rsi')

    ax = plt.subplot(313)
    ax.plot(stok.index.get_values(), stok.tolist(), label=stok.name)
    ax.plot(stod.index.get_values(), stod.tolist(), label=stod.name)
    ax.set_xlabel('t')
    ax.set_ylabel('STO%')

    plt.show()


if __name__ == '__main__':
    visual_techind_test()
